const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;

const Config = extern struct {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
};

const TransformerWeights = struct {
    token_embedding_table: []f32, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: []f32, // (layer, dim) rmsnorm weights
    rms_ffn_weight: []f32, // (layer, dim)
    // weights for matmuls. note dim = n_heads * head_size
    wq: []f32, // (layer, dim, n_heads * head_size)
    wk: []f32, // (layer, dim, n_kv_heads * head_size)
    wv: []f32, // (layer, dim, n_kv_heads * head_size)
    wo: []f32, // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: []f32, // (layer, hidden_dim, dim)
    w2: []f32, // (layer, dim, hidden_dim)
    w3: []f32, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: []f32, // (dim, )
    // (optional) classifier weights for the logits, on the last layer
    wcls: []f32 = undefined,
};

const RunState = struct {
    // current wave of activations
    x: []f32, // activations at current timestamp (dim,)
    xb: []f32, // same, but inside a residual branch (dim,)
    xb2: []f32, // an additional buffer just for convenience (dim, )
    hb: []f32, // buffer for hidden dimensions in the ffn (hidden_dim,)
    hb2: []f32, // buffer for hidden dimensions in the ffn (hidden_dim,)
    q: []f32, // query (dim, )
    k: []f32, // key (dim, )
    v: []f32, // value (dim, )
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits
    // kv cache
    key_cache: []f32, // (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)
};

const Transformer = struct {
    config: Config, // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState, // buffers for the "wave" of activations in the foward pass
    // I might not need these depending on zig differences from c 
    // some more state needed to properly clean up the memory mappings (sigh)
    fd: i32, // file descriptor for memory mappings
    data: []u8, // memory mapped data pointer
    file_size: usize, //size of the checkpoint file in bytes
};

// --------------------------------------------------------------------------
// The Byte Pair Enconding (BPE) Tokenizer that translates strings <-> tokens

const TokenIndex = struct {
    str: []const u8,
    id: i32,
};

const Tokenizer = struct {
    vocab: [][]u8,
    vocab_scores: []f32,
    sorted_vocab: ?[]TokenIndex,
    vocab_size: i32,
    max_token_length: i32,
    byte_pieces: [256]u8, // stores all single-byte strings
};

fn compare_tokens_bsearch(needle: []const u8, a: TokenIndex) std.math.Order {
    return std.mem.order(u8, needle, a.str);
}

fn compare_tokens_less_than(_: @TypeOf(.{}), a: TokenIndex, b: TokenIndex) bool {
    return std.mem.order(u8, a.str, b.str) == .lt;
}

// --------------------------------------------------------------------------
// The sampler, which takes logits and returns a sampled token
// sampling can be done in a few wasy: greedy argmax, sampling top-p sampling

const ProbIndex = struct {
    prob: f32,
    index: i32,

    pub fn more_likely_than(_: @TypeOf(.{}), a: ProbIndex, b: ProbIndex) bool {
        return a.prob > b.prob;
    }
};

const Sampler = struct {
    vocab_size: i32,
    probindex: []ProbIndex, // Buffer used in top-p sampling
    temperature: f32,
    topp: f32,
    seed: u64,
    prng: std.Random.DefaultPrng,
};

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    var checkpoint_path: []const u8 = undefined; // e.g. out/model.bin
    var tokenizer_path: []const u8 = "tokenizer.bin";
    var temperature: f32 = 1.0; // 0.0 = greedy determisitic. 1.0 = original. don't set higher
    var topp: f32 = 0.9; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    var steps: u32 = 256; // number of steps to run for

    var prompt: []const u8 = &.{};
    var seed: u64 = 0; 
    var mode: []const u8 = "generate"; // generate|chat

    var system_prompt: ?[]const u8 = null; // the (optional) system prompt to use in chat mode

    const args = try init.minimal.args.toSlice(arena);
    if (args.len >= 2) {
        checkpoint_path = args[1];
    } else {
        try error_usage(io);
    }

    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) try error_usage(io); // must have arg after flag
        if (args[i][0] != '-') try error_usage(io); // must start with dash
        if (args[i].len != 2) try error_usage(io); // must be -x (one dash, one letter)
        switch (args[i][1]) {
            't' => temperature = std.fmt.parseFloat(f32, args[i+1]) catch return try error_usage(io),
            'p' => topp = std.fmt.parseFloat(f32, args[i+1]) catch return try error_usage(io),
            's' => seed = std.fmt.parseInt(u64, args[i+1], 10) catch return try error_usage(io),
            'n' => steps = std.fmt.parseInt(u32, args[i+1], 10) catch return try error_usage(io),
            'i' => prompt = args[i+1],
            'z' => tokenizer_path = args[i+1],
            'm' => mode = args[i+1],
            'y' => system_prompt = args[i+1],
            else => try error_usage(io),
        }
    }

    if (seed == 0) seed = @intCast(std.Io.Clock.real.now(io).toMilliseconds());
    if (temperature < 0) temperature = 0;
    if (topp < 0 or topp > 1) topp = 0.9;

    // build the transformer via the model .bin file
    var transformer = try build_transformer(io, arena, checkpoint_path);
    const seq_len: u32 = @intCast(transformer.config.seq_len);
    if (steps == 0 or steps > seq_len) {
        steps = seq_len;
    }

    var tokenizer = try build_tokenizer(io, arena, tokenizer_path, transformer.config.vocab_size);
    var sampler = try build_sampler(arena, transformer.config.vocab_size, temperature, topp, seed);

    // Run
    if (std.mem.eql(u8, mode, "generate")) {
        try generate(io, arena, &transformer, &tokenizer, &sampler, prompt, steps);
    } else if (std.mem.eql(u8, mode, "chat")) {

    } else {
        try error_usage(io);
    }
}

fn generate(io: std.Io, alloc: Allocator, transformer: *Transformer, tokenizer: *Tokenizer, sampler: *Sampler, prompt: []const u8, steps: u32) !void {
    // encode the (string) prompt into token sequence
    const tokens_buffer = try alloc.alloc(i32, prompt.len+3);
    const prompt_tokens = try encode(alloc, tokenizer, prompt, true, false, tokens_buffer);
    if (prompt_tokens.len < 1) {
        std.log.err("something went wrong, expected at least 1 prompt token\n", .{});
        return error.NoPromptTokens;
    }

    // start the main loop
    var start: i64 = 0; // used to time our code, only initialized after first iteration
    var next: i32 = undefined; // will store the next token in the sequence
    var token: i32 = prompt_tokens[0];
    var pos: usize = 0; // position in sequence
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        const logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < prompt_tokens.len - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        const piece = decode(tokenizer, token, next);
        safe_print(piece);
        token = next;


        // init the timer here becasuse the first iteration can be slower
        if (start == 0) { 
            start = std.Io.Clock.real.now(io).toMilliseconds();
        }
    }
    std.debug.print("\n", .{});
    // report achived tok/s (pos-1 because the timer starts after the first iteration)
    if (pos > 1) {
        const end = std.Io.Clock.real.now(io).toMilliseconds();
        const posf: f64 = @floatFromInt(pos);
        const startf: f64 = @floatFromInt(start);
        const endf: f64 = @floatFromInt(end);
        std.debug.print("achived tok/s: {}\n", .{(posf - 1) / (endf-startf)*1000});
    }

    // Prompt tokens are never freed but since the program exists I think it's fine, I don't think this will work for chat
    // mode though
}

fn forward(transformer: *Transformer, token: i32, pos: usize) []f32 {
    // a few convenicence variables
    const config = &transformer.config;
    const weights = &transformer.weights;
    const state = &transformer.state;
    const x = state.x;
    const dim: usize = @intCast(config.dim);
    const kv_dim: usize = @intCast(@divFloor(config.dim * config.n_kv_heads, config.n_heads));
    const seq_len: usize = @intCast(config.seq_len);
    const kv_mul: usize = @intCast(@divFloor(config.n_heads, config.n_kv_heads)); // integer multiplier of the kv sharing in multiquery
    const hidden_dim: usize = @intCast(config.hidden_dim);
    const n_heads: usize = @intCast(config.n_heads);
    const head_size: usize = @intCast(@divFloor(dim, n_heads));

    // copy the token embedding into x
    var offset: usize = @as(usize, @intCast(token)) * dim;
    @memcpy(x, weights.token_embedding_table[offset..offset+dim]);

    // forward all the layers
    for (0..@intCast(config.n_layers)) |l| {
        // attention rmsnorm
        offset = l * dim;
        rmsnorm(state.xb, x, weights.rms_att_weight[offset..offset+dim]);

        // key and value point to the kv cache
        const loff = l * seq_len * kv_dim;
        offset = loff+pos*kv_dim;
        state.k = state.key_cache[offset..offset+kv_dim];
        state.v = state.value_cache[offset..offset+kv_dim];

        // qkv matmuls for this position
        offset = l*dim*dim;
        matmul(state.q, state.xb, weights.wq[offset..offset+dim*dim], dim, dim);
        offset = l*dim*kv_dim;
        matmul(state.k, state.xb, weights.wk[offset..offset+dim*kv_dim], dim, kv_dim);
        matmul(state.v, state.xb, weights.wv[offset..offset+dim*kv_dim], dim, kv_dim);

        { // RoPE relative positional encoding: complex-valued rotate q and k each head
            var i: usize = 0;
            while (i < dim) : (i+=2) {
                const head_dim: f32 = @floatFromInt(i % head_size);
                const head_sizef: f32 = @floatFromInt(head_size);
                const freq: f32 = 1.0 / std.math.pow(f32, 10_000, head_dim / head_sizef);
                const val = @as(f32, @floatFromInt(pos)) * freq;
                const fcr = std.math.cos(val);
                const fci = std.math.sin(val);
                const rotn: usize = if (i < kv_dim) 2 else 1;  // how many vectors? 2 = q & k, 1 = q only
                for (0..rotn) |v| {
                    const vec = if (v == 0) state.q else state.k; // the vector to rotate (query or key)
                    const v0 = vec[i];
                    const v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;
                    vec[i+1] = v0 * fci + v1 * fcr;
                }
            }
        }

        // multihead attention. iterate over all heads
        for (0..n_heads) |h| {
            // get the query verctor for this head
            offset = h*head_size;
            const q = state.q[offset..offset+head_size];
            // attention scores for this head 
            offset = h*seq_len;
            const att = state.att[offset..offset+seq_len];
            // interate over all timesteps, including the current one
            for (0..pos+1) |t| {
                // get the key vector for this head an at this timestep
                offset = loff + t * kv_dim + (@divFloor(h, kv_mul)) * head_size;
                const k = state.key_cache[offset..offset+head_size];
                // calculate the attention score as the dot product of q and k
                var score: f32 = 0;
                for (0..head_size) |i| {
                    score += q[i] * k[i];
                }
                score /= std.math.sqrt(@as(f32, @floatFromInt(head_size)));
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, for 0..pos inclusively
            softmax(att[0..pos+1]);

            // weighted sum of the values, store back into xb
            offset = h * head_size;
            const xb = state.xb[offset..offset+head_size];
            @memset(xb, 0);
            for (0..pos+1) |t| {
                // get the value vector for this head and at this timestep
                offset = loff + t * kv_dim + @divFloor(h, kv_mul) * head_size;
                const v = state.value_cache[offset..offset+head_size];
                // get the attention weight for this timestep
                const a = att[t];
                // accumulate the weighted value into xb
                for (0..head_size) |i| {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        offset = l*dim*dim;
        matmul(state.xb2, state.xb, weights.wo[offset..offset+dim*dim], dim, dim);

        // residual connection back into x
        for (0..dim) |i| {
            x[i] += state.xb2[i];
        }

        // ffn rmsnorm
        offset = l*dim;
        rmsnorm(state.xb, x, weights.rms_ffn_weight[offset..offset+dim]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        offset = l*dim*hidden_dim;
        matmul(state.hb, state.xb, weights.w1[offset..offset+dim*hidden_dim], dim, hidden_dim);
        matmul(state.hb2, state.xb, weights.w3[offset..offset+dim*hidden_dim], dim, hidden_dim);

        // SwiGLU non linearity
        for (0..hidden_dim) |i| {
            var val = state.hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1 / (1 + std.math.exp(-val)));
            // elementwise multiply with w3(x)
            val *= state.hb2[i];
            state.hb[i] = val;
        }


        // final matmul to get the output of the ffn
        offset = l*dim*hidden_dim;
        matmul(state.xb, state.hb, weights.w2[offset..offset+hidden_dim*dim], hidden_dim, dim);

        // residual connection
        for (0..dim) |i| {
            x[i] += state.xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, weights.rms_final_weight);

    // classifier into logits
    matmul(state.logits, x, weights.wcls, dim, @intCast(config.vocab_size));

    return state.logits;
}

fn rmsnorm(o: []f32, x: []f32, weights: []f32) void {
    // calculate the sum of squares
    var ss: f32 = 0;
    for (0..weights.len) |i| {
        ss += x[i] * x[i];
    }

    ss /= @floatFromInt(x.len);
    ss += 1e-5;
    ss = 1.0 / std.math.sqrt(ss);
    // normalize and scale
    for (0..weights.len) |i| {
        o[i] = weights[i] * (ss * x[i]);
    }
}

fn matmul(xout: []f32, x: []f32, w: []f32, n: usize, d: usize) void {
    // W (d, n) @ x (n, ) -> xout (d, )
    // by far the most amount of time is spent inside this little function

    // in the karpathy c code he uses omp parallel to make this multi threading, I'm keeping it simple for now
    for (0..d) |i| {
        var val: f32 = 0;
        for (0..n) |j| {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

fn softmax(x: []f32) void {
    // find max value (for numerical stability)
    var max_val: f32 = x[0];
    for (1..x.len) |i| {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exp and sum
    var sum: f32 = 0;
    for (0..x.len) |i| {
        x[i] = std.math.exp(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (0..x.len) |i| {
        x[i] /= sum;
    }
}

fn error_usage(io: std.Io) !void {
    var buff: [4096]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(io, &buff);
    const writer = &stderr.interface;
    try writer.print("Usage:    run <checkpoint> [options]\n", .{});
    try writer.print("Example:  run model.bin -n 256 -i \"Once upon a time\"\n", .{});
    try writer.print("Options:\n", .{});
    try writer.print("  -t <float>  temperature in [0, inf], default 1.0\n", .{});
    try writer.print("  -p <float>  p value in top-p (nucleus) sampling in [0, 1] default 0.9\n", .{});
    try writer.print("  -s <int>    random seed, default time\n", .{});
    try writer.print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n", .{});
    try writer.print("  -i <string> input prompt\n", .{});
    try writer.print("  -z <string> optional path to custom tokenizer\n", .{});
    try writer.print("  -m <string> mode: generate|chat. default generate\n", .{});
    try writer.print("  -y <string> (optional) system prompt in chat mode\n", .{});
    try writer.flush();
    std.process.exit(1);
}


fn build_transformer(io: std.Io, alloc: Allocator, checkpoint_path: []const u8) !Transformer {
    // Read checkpoint file
    var file = std.Io.Dir.cwd().openFile(io, checkpoint_path, .{}) catch |e| {
        std.log.err("Could't open checkpoint file ({s}): {}", .{checkpoint_path, e});
        return error.CouldNotOpenFile;
    };

    var file_buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &file_buffer);
    var reader = &file_reader.interface;

    var config = try reader.takeStruct(Config, .little);

    // negative vocab size is a hacky way of signaling unshared weights. bit yikes.
    const shared_weights = if (config.vocab_size > 0) true else false;
    config.vocab_size = @intCast(@abs(config.vocab_size));

    const file_size = try file_reader.getSize();
    const data = std.posix.mmap(null, file_size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0) catch |e| {
        std.log.err("mmap failed: {}", .{e});
        return error.MMAPFailed;
    };

    const weights_buff: []f32 = @ptrCast(data[@sizeOf(Config)..]);
    const weights = memory_map_weights(config, weights_buff, shared_weights);

    // Allocate the Runstate buffers
    const kv_dim = @divFloor(config.dim * config.n_kv_heads, config.n_heads);
    const state: RunState = .{
        // Maybe all these allocs could be done at once but it's an arena so maybe it doesn't matter???
        .x = try alloc.alloc(f32, @intCast(config.dim)),
        .xb = try alloc.alloc(f32, @intCast(config.dim)),
        .xb2 = try alloc.alloc(f32, @intCast(config.dim)),
        .hb = try alloc.alloc(f32, @intCast(config.hidden_dim)),
        .hb2 = try alloc.alloc(f32, @intCast(config.hidden_dim)),
        .q = try alloc.alloc(f32, @intCast(config.dim)),
        .k = undefined,
        .v = undefined,
        .key_cache = try alloc.alloc(f32, @intCast(config.n_layers * config.seq_len * kv_dim)),
        .value_cache = try alloc.alloc(f32, @intCast(config.n_layers * config.seq_len * kv_dim)),
        .att = try alloc.alloc(f32, @intCast(config.n_heads * config.seq_len)),
        .logits = try alloc.alloc(f32, @intCast(config.vocab_size)),
    };

    return .{
        .config = config,
        .weights = weights,
        .state = state,
        .fd = file.handle,
        .data = data,
        .file_size = file_size,
    };
}

fn memory_map_weights(p: Config, ptr: []f32, shared_weights: bool) TransformerWeights {
    const head_size = @divFloor(p.dim, p.n_heads);
    const n_layers = p.n_layers;

    var start: usize = 0;
    var end: usize = @intCast(p.vocab_size * p.dim);
    var w: TransformerWeights = undefined;
    w.token_embedding_table = ptr[start..end];
    start = end; end += @intCast(n_layers * p.dim);
    w.rms_att_weight = ptr[start..end];
    start = end; end += @intCast(n_layers * p.dim * (p.n_heads * head_size));
    w.wq = ptr[start..end];
    start = end; end += @intCast(n_layers * p.dim * (p.n_kv_heads * head_size));
    w.wk = ptr[start..end];
    start = end; end += @intCast(n_layers * p.dim * (p.n_kv_heads * head_size));
    w.wv = ptr[start..end];
    start = end; end += @intCast(n_layers * (p.n_heads * head_size) * p.dim);
    w.wo = ptr[start..end];
    start = end; end += @intCast(n_layers * p.dim);
    w.rms_ffn_weight = ptr[start..end];
    start = end; end += @intCast(n_layers * p.dim * p.hidden_dim);
    w.w1 = ptr[start..end];
    start = end; end += @intCast(n_layers * p.hidden_dim * p.dim);
    w.w2 = ptr[start..end];
    start = end; end += @intCast(n_layers * p.dim * p.hidden_dim);
    w.w3 = ptr[start..end];
    start = end; end += @intCast(p.dim);
    w.rms_final_weight = ptr[start..end];
    end += @intCast(p.seq_len * @divFloor(head_size, 2)); // skip what used to be freq_cis_real (for RoPE)
    end += @intCast(p.seq_len * @divFloor(head_size, 2)); // skip what used to be freq_cis_imag (for RoPE)
    start = end; end = ptr.len;
    w.wcls = if (shared_weights) w.token_embedding_table else ptr[start..end];
    return w;
}

fn build_tokenizer(io: std.Io, alloc: Allocator, tokenizer_path: []const u8, vocab_size: i32) !Tokenizer {
    var t: Tokenizer = undefined;
    // karpathy: i should have written the vocab_size into the tokenizer file... sigh
    t.vocab_size = vocab_size;
    // alloc space to hold the scores and the strings
    t.vocab = try alloc.alloc([]u8, @intCast(vocab_size));
    t.vocab_scores = try alloc.alloc(f32, @intCast(vocab_size));
    t.sorted_vocab = null; // initialized lazily
    for (0..256) |i| {
        t.byte_pieces[i] = @intCast(i);
    }

    // read in file
    const file = try std.Io.Dir.cwd().openFile(io, tokenizer_path, .{});
    defer file.close(io);
    var file_buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &file_buffer);
    var reader = &file_reader.interface;

    t.max_token_length = try reader.takeInt(i32, .little);
    var len: i32 = 0;
    for (0..@intCast(vocab_size)) |i| {
        const score_bytes = try reader.takeInt(u32, .little); // u32 is the same size as f32
        t.vocab_scores[i] = @bitCast(score_bytes);
        len = try reader.takeInt(i32, .little);
        t.vocab[i] = try alloc.alloc(u8, @intCast(len));
        try reader.readSliceAll(t.vocab[i]);
    }

    return t;
}

fn build_sampler(alloc: Allocator, vocab_size: i32, temperature: f32, topp: f32, seed: u64) !Sampler {
    return .{
        .vocab_size = vocab_size,
        .temperature = temperature,
        .topp = topp,
        .seed = seed,
        .prng = std.Random.DefaultPrng.init(seed),
        .probindex = try alloc.alloc(ProbIndex, @intCast(vocab_size)),
    };
}

fn encode(alloc: Allocator, t: *Tokenizer, text: []const u8, bos: bool, eos: bool, tokens: []i32) ![]i32 {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (t.sorted_vocab == null) {
        t.sorted_vocab = try alloc.alloc(TokenIndex, @intCast(t.vocab_size));
        for (0..@intCast(t.vocab_size)) |i| {
            t.sorted_vocab.?[i].str = t.vocab[i];
            t.sorted_vocab.?[i].id = @intCast(i);
        }
        std.sort.block(TokenIndex, t.sorted_vocab.?, .{}, compare_tokens_less_than);
    }

    // start at 0 tokens
    var n_tokens: usize = 0;

    // add optional BOS (=1) token, if desired
    if (bos) {
        tokens[n_tokens] = 1;
        n_tokens += 1;
    }

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // karpathy: TODO: pretty sure this isn't corrent in the general case but I don't have the
    // energy to read more of the sentecepiece code to figure out what it's doing
    if (text.len != 0) {
        tokens[n_tokens] = str_lookup(" ", t.sorted_vocab.?).?;
        n_tokens += 1;
    }

    // Using zigs std library for UTF-8 parsing. karpathy did it manually but this seems less 
    // error prone for me.
    var view = try std.unicode.Utf8View.init(text);
    var it = view.iterator();
    while (it.nextCodepointSlice()) |codepoint| {
        const maybe = str_lookup(codepoint, t.sorted_vocab.?);
        if (maybe) |id| {
            tokens[n_tokens] = id;
            n_tokens += 1;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at 3
            for (0..codepoint.len) |i| {
                tokens[n_tokens] = @as(i32, codepoint[i]) + 3;
                n_tokens += 1;
            }
        }
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +2 for UTF8 (in case max_token_length is 1)
    const str_buffer = try alloc.alloc(u8, @intCast(t.max_token_length*2 + 2));

    // merge the best consecutive pair each iteration, according to the scores in vocab_scores
    while (true) {
        var best_score: f32 = -1e10;
        var best_id: i32 = -1;
        var best_idx: i32 = -1;

        for (0..n_tokens-|1) |i| {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            const curr = tokens[i];
            const next = tokens[i+1];
            // const merged = try std.fmt.bufPrint(str_buffer, "{s}{s}", .{t.vocab[@intCast(tokens[i])], t.vocab[@intCast(tokens[i+1])]});
            const merged = try std.fmt.bufPrint(str_buffer, "{s}{s}", .{t.vocab[@intCast(curr)], t.vocab[@intCast(next)]});
            const id = str_lookup(merged, t.sorted_vocab.?);
            if (id != null and t.vocab_scores[@intCast(id.?)] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t.vocab_scores[@intCast(id.?)];
                best_id = id.?;
                best_idx = @intCast(i);
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[@intCast(best_idx)] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (@intCast(best_idx+1)..n_tokens-1) |i| {
            tokens[i] = tokens[i+1];
        }

        n_tokens -= 1;
    }

    // add optional EOS (=2) token, if desired
    if (eos) {
        tokens[n_tokens] = 2;
        n_tokens += 1;
    }

    return tokens[0..n_tokens];
}

fn decode(t: *Tokenizer, prev_token: i32, token: i32) []const u8 {
    var piece = t.vocab[@intCast(token)];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespaces
    if (prev_token == 1 and piece.len > 0 and piece[0] == ' ') {
        piece = piece[1..];
    }
    // careful, some tokens designate raw bytes, and look like e.g '<0x01>'
    // parse this and convert and return the actual byte
    if (piece.len == 6 and std.mem.startsWith(u8, piece, "<0x") and piece[5] == '>') {
        if (std.fmt.parseInt(u8, piece[3..5], 16)) |byte_piece| {
            piece = t.byte_pieces[byte_piece..byte_piece+1];
        } else |_| {} // not valid hex keep as is
    }

    return piece;
}

fn str_lookup(str: []const u8, sorted_vocab: []TokenIndex) ?i32 {
    // efficiently fin the prefect match for str in vocab, return it's index or -1 if not found
    const res = std.sort.binarySearch(TokenIndex, sorted_vocab, str, compare_tokens_bsearch);
    return if (res != null) sorted_vocab[res.?].id else null;
}

fn sample(sampler: *Sampler, logits: []f32) i32 {
    // sample the token given the logits and some hyperparameters
    var next: i32 = undefined;
    if (sampler.temperature == 0) {
        // greedy argmax sampling: take the token with the highest probablility
        next = @intCast(sample_argmax(logits));
    } else {
        // apply the temperature to the logits
        for (0..logits.len) |q| {
            logits[q] /= sampler.temperature;
        }
        // apply softmax to the logits to get the probabilities for the next token
        softmax(logits);
        // flip a (float) coin (this is our source of entropy for sampling)
        const coin = sampler.prng.random().float(f32);
        // we sample form this distribution to get the next token 
        if (sampler.topp <= 0 or sampler.topp >= 1) {
            // simply sample from the predicted probability distribution
            next = @intCast(sample_mult(logits, coin));
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = @intCast(sample_topp(logits, sampler.topp, sampler.probindex, coin));
        }
    }

    return next;
}

fn sample_argmax(probabilities: []f32) usize {
    // return the max index that has the highest probability
    var max_i: usize = 0;
    var max_p = probabilities[0];
    for (1..probabilities.len) |i| {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }

    return max_i;
}

fn sample_mult(probabilities: []f32, coin: f32) usize {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number [0, 1)
    var cdf: f32 = 0.0; 
    for (0..probabilities.len) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }

    return probabilities.len - 1; // in case of rounding error
}

fn sample_topp(probabilities: []f32, topp: f32, probindex: []ProbIndex, coin: f32) usize {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1).

    var n0: usize = 0;
    // sort indices in deceinding order of probabilities
    // values smaller than (1 - topp) / (len - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const n: f32 = @floatFromInt(probabilities.len);
    const cutoff = (1 - topp) / (n - 1);
    for (0..probabilities.len) |i| {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = @intCast(i);
            probindex[n0].prob = probabilities[i];
            n0 += 1;
        }
    }

    std.sort.block(ProbIndex, probindex[0..n0], .{}, ProbIndex.more_likely_than);

    // truncate the list where cumulative probability exceeds topp
    var cumulative_prob: f32 = 0.0;
    var last_idx = n0 - 1;  // in case of rounding errors consider all elements;
    for (0..n0) |i| {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    const r = coin * cumulative_prob;
    var cdf: f32 = 0;
    for (0..last_idx+1) |i| {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return @intCast(probindex[i].index);
        }
    }

    return @intCast(probindex[last_idx].index); // in case of rounding errors
}

fn safe_print(piece: []const u8) void {
    if (piece.len == 1) {
        if (!std.ascii.isPrint(piece[0]) and !std.ascii.isWhitespace(piece[0])) {
            return;
        }
    }

    std.debug.print("{s}", .{piece});
}
