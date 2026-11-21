# NInfer Jinja source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`7609846557c50f9d984719a9e1e8c5f3d02f807b`](https://github.com/ggml-org/llama.cpp/commit/7609846557c50f9d984719a9e1e8c5f3d02f807b),
`common/jinja/`, under the MIT license; see [LICENSE](LICENSE).

NInfer maintains this fork for chat-template rendering and adopts upstream fixes selectively.

Unicode case data is generated from Unicode 17.0.0 `SpecialCasing.txt` and
`DerivedCoreProperties.txt`, under [UNICODE-LICENSE](UNICODE-LICENSE):

```bash
python3 tools/generate_jinja_unicode.py /path/to/unicode-17.0.0
```

## ingine

ingine vendors this runtime from fngine. The only change: `jinja/unicode.cpp` takes its UTF-8 helpers from the
bundled `jinja/utf8_support.{h,cpp}` (copied from fngine `src/text/unicode.cpp`) instead of the engine's
`text/unicode.h`, so the `ingine_jinja` target builds on its own.
`jinja/json.cpp` formats finite floats like Python's `repr()` (`python_float_repr`), as the reference chat
rendering does, instead of nlohmann's dump (which switches to exponent notation at 1e15).
