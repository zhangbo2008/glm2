import transformers.utils.import_utils as f
a=f.is_accelerate_available() and f.is_bitsandbytes_available()
print(a)
print((is_accelerate_available() and is_bitsandbytes_available()))