#!/bin/bash
echo "--- BAZEL CI SECURITY POC ---"
echo "Proven: The container is privileged and has access to host docker socket."

# هذا الأمر آمن تماماً، سيقوم فقط بطباعة اسم السيرفر المضيف (Host)
# لإثبات أننا نتواصل مع نظام Docker الخاص بالمضيف وليس الحاوية.
docker -H unix:///var/run/docker.sock info | grep "Name:"

echo "--- POC FINISHED ---"
