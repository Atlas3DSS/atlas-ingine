// Token-level logit bitmask interface for constrained decoding (CUDA-free).
#pragma once

#include <cstddef>
#include <cstdint>

namespace ingine::model {

// Stateful per-request token masker (e.g. JSON / JSON-Schema pushdown automaton).
// Callers check is_active() before allocating/uploading a mask so unconstrained phases
// (e.g. reasoning/thinking channels) incur zero mask computation or GPU overhead.
class TokenMasker {
 public:
  virtual ~TokenMasker() = default;

  // Vocabulary size V covered by fill_mask() (bit v in [0, V) => word v/32, bit v%32).
  virtual int vocab_size() const = 0;

  // Number of 32-bit words needed to hold V bits.
  std::size_t mask_words() const { return (std::size_t(vocab_size()) + 31u) / 32u; }

  // True when logits must be masked on the upcoming sampling step.
  virtual bool is_active() const = 0;

  // Writes packed bitmask (bit v == 1 => token v allowed, bit v == 0 => masked to -inf).
  // Returns false if no masking is needed on this step.
  virtual bool fill_mask(std::uint32_t* words, std::size_t word_count) const = 0;

  // Advances internal state with the sampled/accepted token id.
  virtual void advance(std::int32_t token_id) = 0;
};

}  // namespace ingine::model
