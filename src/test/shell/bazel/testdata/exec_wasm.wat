(module
  (memory 1)

  (func $allocate (export "allocate")
    (param $size i32)
    (param $align i32)
    (result i32)
    (local $orig_page_count i32)

    ;; Pass each allocation directly to `memory.grow`, which allocates the
    ;; requested number of pages (each page is 65536 bytes).

    ;; PAGE_SIZE = 65536
    ;; page_count = min(1, $size % PAGE_SIZE) + ($size / PAGE_SIZE)
    local.get $size
    i32.const 65536
    i32.rem_u
    (if (result i32)
      (then i32.const 1)
      (else i32.const 0))
    local.get $size
    i32.const 65536
    i32.div_u
    i32.add

    ;; If allocation is successful then the output pointer will be the original
    ;; heap size in bytes. If not successful, the pushed value will be -1.
    memory.grow
    local.tee $orig_page_count
    i32.const -1
    i32.eq
    (if (result i32)
      (then i32.const 0)
      (else
        local.get $orig_page_count
        i32.const 65536
        i32.mul
      )
    )
    return
  )

  ;; Returns the output buffer "ok" with return code 0.
  (func (export "run_ok")
    (param $input_ptr i32)
    (param $input_len i32)
    (param $output_ptr_ptr i32)
    (param $output_len_ptr i32)
    (result i32)
    (local $output_ptr i32)

    ;; *output_len_ptr = 2
    local.get $output_len_ptr
    i32.const 2
    i32.store

    ;; output_ptr = allocate(size=2, align=1)
    i32.const 2
    i32.const 1
    call $allocate
    local.set $output_ptr

    ;; *output_ptr_ptr = output_ptr
    local.get $output_ptr_ptr
    local.get $output_ptr
    i32.store

    ;; output_ptr[0] = b"o"
    local.get $output_ptr
    i32.const 0x6F
    i32.store8

    ;; output_ptr[1] = b"k"
    local.get $output_ptr
    i32.const 1
    i32.add
    i32.const 0x6B
    i32.store8

    ;; return 0
    i32.const 0
    return
  )

  ;; Returns the output buffer "err" with return code 1.
  (func (export "run_err")
    (param $input_ptr i32)
    (param $input_len i32)
    (param $output_ptr_ptr i32)
    (param $output_len_ptr i32)
    (result i32)
    (local $output_ptr i32)

    ;; *output_len_ptr = 3
    local.get $output_len_ptr
    i32.const 3
    i32.store

    ;; output_ptr = allocate(size=3, align=1)
    i32.const 3
    i32.const 1
    call $allocate
    local.set $output_ptr

    ;; *output_ptr_ptr = output_ptr
    local.get $output_ptr_ptr
    local.get $output_ptr
    i32.store

    ;; output_ptr[0] = b"e"
    local.get $output_ptr
    i32.const 0x65
    i32.store8

    ;; output_ptr[1] = b"r"
    local.get $output_ptr
    i32.const 1
    i32.add
    i32.const 0x72
    i32.store8

    ;; output_ptr[2] = b"r"
    local.get $output_ptr
    i32.const 2
    i32.add
    i32.const 0x72
    i32.store8

    ;; return 1
    i32.const 1
    return
  )

  ;; Reports an output length far larger than the module could ever allocate,
  ;; without allocating a buffer to match it. Used by
  ;; test_execute_wasm_output_len_exceeds_memory_limit.
  ;;
  ;; The host reads the length back out of guest memory and uses it to size a
  ;; host-side buffer, so a module that lies about it decides how much host
  ;; memory gets allocated. The length is left as i32 max, which is the value
  ;; that reached the JVM allocator before the host bounded it.
  (func (export "run_overflow_len")
    (param $input_ptr i32)
    (param $input_len i32)
    (param $output_ptr_ptr i32)
    (param $output_len_ptr i32)
    (result i32)

    ;; *output_len_ptr = 0x7FFFFFFF
    local.get $output_len_ptr
    i32.const 0x7FFFFFFF
    i32.store

    ;; *output_ptr_ptr = 0
    ;;
    ;; Deliberately null: the host must reject the length before it dereferences
    ;; the pointer, so this test fails loudly if the two checks are ever
    ;; reordered.
    local.get $output_ptr_ptr
    i32.const 0
    i32.store

    ;; return 0
    i32.const 0
    return
  )
)
