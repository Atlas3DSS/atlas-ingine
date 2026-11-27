#include "backends/qwen4exp/qwen4exp_config.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ingine::backends::qwen4exp {

namespace {

std::int64_t checked_mul_pos_i32(int a, int b, const char* what) {
  const std::int64_t prod = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
  if (prod <= 0 || prod > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("Qwen4ExpConfig: integer overflow in ") + what);
  }
  return prod;
}

}  // namespace

const char* to_string(Qwen4ExpLayerKind kind) noexcept {
  switch (kind) {
    case Qwen4ExpLayerKind::LinearAttention: return "linear_attention";
    case Qwen4ExpLayerKind::FullAttention: return "full_attention";
  }
  return "unknown";
}

void Qwen4ExpConfig::finalize_and_validate() {
  if (architecture != "qwen4exp") {
    throw std::invalid_argument("Qwen4ExpConfig requires architecture == 'qwen4exp', got '" + architecture + "'");
  }
  if (vocab_size <= 0) throw std::invalid_argument("Qwen4ExpConfig: vocab_size must be positive");
  if (hidden_size <= 0) throw std::invalid_argument("Qwen4ExpConfig: hidden_size must be positive");
  if (num_hidden_layers <= 0) throw std::invalid_argument("Qwen4ExpConfig: num_hidden_layers must be positive");
  if (max_position_embeddings <= 0) {
    throw std::invalid_argument("Qwen4ExpConfig: max_position_embeddings must be positive");
  }
  if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0f) {
    throw std::invalid_argument("Qwen4ExpConfig: rms_norm_eps must be finite and positive");
  }
  if (hc_mult <= 0 || hc_low_rank <= 0) {
    throw std::invalid_argument("Qwen4ExpConfig: hyper_connection count and low_rank must be positive");
  }
  (void)checked_mul_pos_i32(hc_mult, hidden_size, "hc_dim");
  if (num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0) {
    throw std::invalid_argument("Qwen4ExpConfig: invalid full-attention dimensions");
  }
  const std::int64_t attn_dim = checked_mul_pos_i32(num_attention_heads, head_dim, "num_attention_heads * head_dim");
  if (attn_dim > std::numeric_limits<int>::max() / 2) {
    throw std::invalid_argument("Qwen4ExpConfig: integer overflow in full_q_out_dim");
  }
  (void)checked_mul_pos_i32(num_key_value_heads, head_dim, "full_kv_out_dim");
  if (num_attention_heads % num_key_value_heads != 0) {
    throw std::invalid_argument("Qwen4ExpConfig: num_attention_heads must be divisible by num_key_value_heads");
  }
  if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0) {
    throw std::invalid_argument("Qwen4ExpConfig: rotary_dim must be even and in (0, head_dim]");
  }
  if (!std::isfinite(rope_theta) || rope_theta <= 0.0f) {
    throw std::invalid_argument("Qwen4ExpConfig: rope_theta must be positive");
  }
  if (ssm_conv_kernel < 2 || ssm_state_size <= 0 || ssm_group_count <= 0 || ssm_time_step_rank <= 0) {
    throw std::invalid_argument("Qwen4ExpConfig: invalid Gated DeltaNet SSM dimensions");
  }
  if (ssm_time_step_rank % ssm_group_count != 0) {
    throw std::invalid_argument("Qwen4ExpConfig: ssm_time_step_rank must be divisible by ssm_group_count");
  }
  const std::int64_t gdn_k = checked_mul_pos_i32(ssm_group_count, ssm_state_size, "gdn_key_dim");
  const std::int64_t gdn_v = checked_mul_pos_i32(ssm_time_step_rank, ssm_state_size, "gdn_value_dim");
  if (gdn_k * 2 + gdn_v > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Qwen4ExpConfig: integer overflow in gdn_conv_dim");
  }
  if (static_cast<std::int64_t>(ssm_inner_size) != gdn_v) {
    throw std::invalid_argument("Qwen4ExpConfig: ssm_inner_size must equal ssm_time_step_rank * ssm_state_size");
  }
  if (expert_count <= 0 || expert_used_count <= 0 || expert_used_count > expert_count || expert_ffn_length <= 0) {
    throw std::invalid_argument("Qwen4ExpConfig: invalid MoE expert parameters");
  }
  if (shared_expert_ffn_length < 0) {
    throw std::invalid_argument("Qwen4ExpConfig: shared_expert_ffn_length cannot be negative");
  }
  if (!std::isfinite(expert_weights_scale) || expert_weights_scale <= 0.0f) {
    throw std::invalid_argument("Qwen4ExpConfig: expert_weights_scale must be finite and positive");
  }

  // Derive hybrid layer schedule if caller did not populate layer_types explicitly.
  if (layer_types.empty()) {
    if (full_attention_interval <= 0) {
      throw std::invalid_argument("Qwen4ExpConfig: full_attention_interval must be positive");
    }
    layer_types.resize(static_cast<std::size_t>(num_hidden_layers));
    for (int i = 0; i < num_hidden_layers; ++i) {
      layer_types[static_cast<std::size_t>(i)] =
          ((i + 1) % full_attention_interval == 0) ? Qwen4ExpLayerKind::FullAttention
                                                   : Qwen4ExpLayerKind::LinearAttention;
    }
  } else if (static_cast<int>(layer_types.size()) != num_hidden_layers) {
    throw std::invalid_argument("Qwen4ExpConfig: layer_types length must equal num_hidden_layers");
  }

  num_linear_layers = 0;
  num_full_layers = 0;
  for (Qwen4ExpLayerKind k : layer_types) {
    if (k == Qwen4ExpLayerKind::LinearAttention) ++num_linear_layers;
    else ++num_full_layers;
  }

  if (compress_ratios.empty()) {
    compress_ratios.assign(static_cast<std::size_t>(num_hidden_layers), 0);
  } else if (static_cast<int>(compress_ratios.size()) != num_hidden_layers) {
    throw std::invalid_argument("Qwen4ExpConfig: compress_ratios length must equal num_hidden_layers");
  }
  for (int i = 0; i < num_hidden_layers; ++i) {
    const int r = compress_ratios[static_cast<std::size_t>(i)];
    if (r < 0) throw std::invalid_argument("Qwen4ExpConfig: compress_ratios cannot be negative");
    if (r > 0) {
      if (layer_types[static_cast<std::size_t>(i)] != Qwen4ExpLayerKind::FullAttention) {
        throw std::invalid_argument("Qwen4ExpConfig: compress_ratio > 0 only valid on full-attention layers");
      }
      if (indexer_head_count <= 0 || indexer_key_length <= 0 || indexer_top_k <= 0 || rotary_dim > indexer_key_length) {
        throw std::invalid_argument("Qwen4ExpConfig: QSA compress_ratio > 0 requires valid indexer dimensions");
      }
      (void)checked_mul_pos_i32(indexer_head_count, indexer_key_length, "indexer_head_count * indexer_key_length");
    }
  }

  is_ple_layer.assign(static_cast<std::size_t>(num_hidden_layers), 0);
  if (!ple_layers.empty()) {
    for (int il : ple_layers) {
      if (il < 0 || il >= num_hidden_layers) {
        throw std::invalid_argument("Qwen4ExpConfig: ple_layers entry out of range");
      }
      is_ple_layer[static_cast<std::size_t>(il)] = 1;
    }
    if (ple_ngram_size < 2 || ple_ngram_size > 8 || ple_heads_per_ngram <= 0 || ple_conv_kernel < 1 ||
        ple_head_dim <= 0) {
      throw std::invalid_argument("Qwen4ExpConfig: invalid PLE hyperparameters");
    }
    const int n_heads = ple_n_heads();
    if (n_heads <= 0 || n_heads > 64) {
      throw std::invalid_argument("Qwen4ExpConfig: ple_n_heads out of range [1, 64]");
    }
    if (checked_mul_pos_i32(n_heads, ple_head_dim, "ple_n_heads * ple_head_dim") != hidden_size) {
      throw std::invalid_argument("Qwen4ExpConfig: ple_n_heads * ple_head_dim must equal hidden_size");
    }
    if (static_cast<int>(ple_layer_multipliers.size()) < ple_ngram_size) {
      throw std::invalid_argument("Qwen4ExpConfig: ple_layer_multipliers must have at least ple_ngram_size entries");
    }
    if (static_cast<int>(ple_head_offsets.size()) != n_heads ||
        static_cast<int>(ple_head_vocab_sizes.size()) != n_heads) {
      throw std::invalid_argument("Qwen4ExpConfig: ple_head_offsets and ple_head_vocab_sizes must equal ple_n_heads");
    }
    for (int h = 0; h < n_heads; ++h) {
      if (ple_head_vocab_sizes[static_cast<std::size_t>(h)] == 0) {
        throw std::invalid_argument("Qwen4ExpConfig: ple_head_vocab_sizes entry cannot be 0");
      }
    }
  }
}

std::string Qwen4ExpConfig::canonical_fingerprint() const {
  std::ostringstream os;
  os << "arch=" << architecture << ";V=" << vocab_size << ";H=" << hidden_size << ";L=" << num_hidden_layers
     << ";ctx=" << max_position_embeddings << ";hc=" << hc_mult << "x" << hc_low_rank
     << ";attn=" << num_attention_heads << "/" << num_key_value_heads << "x" << head_dim << ":rot=" << rotary_dim
     << ";idx=" << indexer_head_count << "x" << indexer_key_length << "@" << indexer_top_k
     << ";gdn=" << ssm_group_count << "/" << ssm_time_step_rank << "x" << ssm_state_size << ":k=" << ssm_conv_kernel
     << ";moe=" << expert_count << "x" << expert_used_count << ":" << expert_ffn_length << "+"
     << shared_expert_ffn_length << ";ple=" << ple_layers.size() << ":" << ple_ngram_size << "x" << ple_heads_per_ngram
     << "x" << ple_head_dim;
  return os.str();
}

}  // namespace ingine::backends::qwen4exp
