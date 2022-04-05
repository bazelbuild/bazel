if [[ -z "$RELEASE_NAME" ]]; then
  echo BUILD_SCM_REVISION UNSAFE_"$(git rev-parse --abbrev-ref HEAD)"
else
  echo "BUILD_SCM_REVISION $RELEASE_NAME"
fi
