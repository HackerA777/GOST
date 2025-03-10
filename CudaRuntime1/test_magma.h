#pragma once
#include <iostream>
#include <openssl/err.h>

/* For controlled success */
#define T(e)                                    \
  if (!(e)) {                                   \
    ERR_print_errors_fp(stderr);                \
    OPENSSL_die(#e, __FILE__, __LINE__);        \
  }
/* For controlled failure */
#define TF(e)                                   \
  if ((e)) {                                    \
    ERR_print_errors_fp(stderr);                \
  } else {                                      \
    OPENSSL_die(#e, __FILE__, __LINE__);        \
  }

#define TEST_ASSERT(e)                                  \
  {                                                     \
    if (!(test = (e)))                                  \
      printf(cRED "  Test FAILED" cNORM "\n");          \
    else                                                \
      printf(cGREEN "  Test passed" cNORM "\n");        \
  }

void hexdump(const void* ptr, size_t len);