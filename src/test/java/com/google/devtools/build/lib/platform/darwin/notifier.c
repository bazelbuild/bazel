#include <notify.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

// Takes the arg and a state and posts it as a notification.
int main(int argc, const char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "error: usage is `notifier <notification> <state>`\n");
    return EXIT_FAILURE;
  }
  int token;
  uint32_t status = notify_register_check(argv[1], &token);
  if (status != NOTIFY_STATUS_OK) {
    fprintf(stderr, "error: notify_register_check failed: %s (%d)\n", argv[1],
            status);
    return EXIT_FAILURE;
  }
  char *endptr;
  int64_t state = strtoll(argv[2], &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "error: unable to convert: %s to integer\n", argv[2]);
    return EXIT_FAILURE;
  }
  status = notify_set_state(token, state);
  if (status != NOTIFY_STATUS_OK) {
    fprintf(stderr, "error: notify_set_state failed: %s state:%lld (%d)\n",
            argv[1], state, status);
    return EXIT_FAILURE;
  }
  status = notify_post(argv[1]);
  if (status != NOTIFY_STATUS_OK) {
    fprintf(stderr, "error: notify_post failed: %s state:%lld (%d)\n", argv[1],
            state, status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
