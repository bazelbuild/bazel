/* Copyright (c) 2014, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 201410L
#endif

#include <openssl/crypto.h>
#include <openssl/lhash.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct dummy_lhash_node {
  char *s;
  struct dummy_lhash_node *next;
};

struct dummy_lhash {
  struct dummy_lhash_node *head;
};

static void dummy_lh_free(struct dummy_lhash *lh) {
  struct dummy_lhash_node *cur, *next;

  for (cur = lh->head; cur != NULL; cur = next) {
    next = cur->next;
    free(cur->s);
    free(cur);
  }
}

static size_t dummy_lh_num_items(const struct dummy_lhash *lh) {
  size_t count = 0;
  struct dummy_lhash_node *cur;

  for (cur = lh->head; cur != NULL; cur = cur->next) {
    count++;
  }

  return count;
}

static char *dummy_lh_retrieve(struct dummy_lhash *lh, const char *s) {
  struct dummy_lhash_node *cur;

  for (cur = lh->head; cur != NULL; cur = cur->next) {
    if (strcmp(cur->s, s) == 0) {
      return cur->s;
    }
  }

  return NULL;
}

static int dummy_lh_insert(struct dummy_lhash *lh, char **old_data, char *s) {
  struct dummy_lhash_node *node, *cur;

  for (cur = lh->head; cur != NULL; cur = cur->next) {
    if (strcmp(cur->s, s) == 0) {
      *old_data = cur->s;
      cur->s = s;
      return 1;
    }
  }

  node = malloc(sizeof(struct dummy_lhash_node));
  *old_data = NULL;
  node->s = s;
  node->next = lh->head;
  lh->head = node;
  return 1;
}

static char *dummy_lh_delete(struct dummy_lhash *lh, const void *s) {
  struct dummy_lhash_node *cur, **next_ptr;
  char *ret;

  next_ptr = &lh->head;
  for (cur = lh->head; cur != NULL; cur = cur->next) {
    if (strcmp(cur->s, s) == 0) {
      ret = cur->s;
      *next_ptr = cur->next;
      free(cur);
      return ret;
    }
    next_ptr = &cur->next;
  }

  return NULL;
}

static char *rand_string(void) {
  unsigned len = 1 + (rand() % 3);
  char *ret = malloc(len + 1);
  unsigned i;

  for (i = 0; i < len; i++) {
    ret[i] = '0' + (rand() & 7);
  }
  ret[i] = 0;

  return ret;
}

int main(int argc, char **argv) {
  _LHASH *lh;
  struct dummy_lhash dummy_lh = {NULL};
  unsigned i;

  CRYPTO_library_init();

  lh = lh_new(NULL, NULL);
  if (lh == NULL) {
    return 1;
  }

  for (i = 0; i < 100000; i++) {
    unsigned action;
    char *s, *s1, *s2;

    if (dummy_lh_num_items(&dummy_lh) != lh_num_items(lh)) {
      fprintf(stderr, "Length mismatch\n");
      return 1;
    }

    action = rand() % 3;
    switch (action) {
      case 0:
        s = rand_string();
        s1 = (char *)lh_retrieve(lh, s);
        s2 = dummy_lh_retrieve(&dummy_lh, s);
        if (s1 != NULL && (s2 == NULL || strcmp(s1, s2) != 0)) {
          fprintf(stderr, "lh_retrieve failure\n");
          abort();
        }
        free(s);
        break;

      case 1:
        s = rand_string();
        lh_insert(lh, (void **)&s1, s);
        dummy_lh_insert(&dummy_lh, &s2, strdup(s));

        if (s1 != NULL && (s2 == NULL || strcmp(s1, s2) != 0)) {
          fprintf(stderr, "lh_insert failure\n");
          abort();
        }

        if (s1) {
          free(s1);
        }
        if (s2) {
          free(s2);
        }
        break;

      case 2:
        s = rand_string();
        s1 = lh_delete(lh, s);
        s2 = dummy_lh_delete(&dummy_lh, s);

        if (s1 != NULL && (s2 == NULL || strcmp(s1, s2) != 0)) {
          fprintf(stderr, "lh_insert failure\n");
          abort();
        }

        if (s1) {
          free(s1);
        }
        if (s2) {
          free(s2);
        }
        free(s);
        break;

      default:
        abort();
    }
  }

  lh_doall(lh, free);
  lh_free(lh);
  dummy_lh_free(&dummy_lh);
  printf("PASS\n");
  return 0;
}
