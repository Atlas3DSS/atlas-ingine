# Text goldens (CPU): tokenizer, chat renders, StreamParser, Python-compat JSON helpers.
ingine_add_test(NAME text_tokenizer SOURCES test_tokenizer.cpp LIBS ingine_text TIMEOUT 300)
ingine_add_test(NAME text_chat SOURCES test_chat.cpp LIBS ingine_text TIMEOUT 300)
ingine_add_test(NAME text_stream SOURCES test_stream.cpp LIBS ingine_text TIMEOUT 300)
ingine_add_test(NAME text_pyjson SOURCES test_pyjson.cpp LIBS ingine_text)
