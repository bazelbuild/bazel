// Copyright 2016 Google Inc. All Rights Reserved.
//
// Distributed under MIT license.
// See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

package cbrotli

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"testing"
	"time"
)

func checkCompressedData(compressedData, wantOriginalData []byte) error {
	uncompressed, err := Decode(compressedData)
	if err != nil {
		return fmt.Errorf("brotli decompress failed: %v", err)
	}
	if !bytes.Equal(uncompressed, wantOriginalData) {
		if len(wantOriginalData) != len(uncompressed) {
			return fmt.Errorf(""+
				"Data doesn't uncompress to the original value.\n"+
				"Length of original: %v\n"+
				"Length of uncompressed: %v",
				len(wantOriginalData), len(uncompressed))
		}
		for i := range wantOriginalData {
			if wantOriginalData[i] != uncompressed[i] {
				return fmt.Errorf(""+
					"Data doesn't uncompress to the original value.\n"+
					"Original at %v is %v\n"+
					"Uncompressed at %v is %v",
					i, wantOriginalData[i], i, uncompressed[i])
			}
		}
	}
	return nil
}

func TestEncoderNoWrite(t *testing.T) {
	out := bytes.Buffer{}
	e := NewWriter(&out, WriterOptions{Quality: 5})
	if err := e.Close(); err != nil {
		t.Errorf("Close()=%v, want nil", err)
	}
	// Check Write after close.
	if _, err := e.Write([]byte("hi")); err == nil {
		t.Errorf("No error after Close() + Write()")
	}
}

func TestEncoderEmptyWrite(t *testing.T) {
	out := bytes.Buffer{}
	e := NewWriter(&out, WriterOptions{Quality: 5})
	n, err := e.Write([]byte(""))
	if n != 0 || err != nil {
		t.Errorf("Write()=%v,%v, want 0, nil", n, err)
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close()=%v, want nil", err)
	}
}

func TestWriter(t *testing.T) {
	// Test basic encoder usage.
	input := []byte("<html><body><H1>Hello world</H1></body></html>")
	out := bytes.Buffer{}
	e := NewWriter(&out, WriterOptions{Quality: 1})
	in := bytes.NewReader([]byte(input))
	n, err := io.Copy(e, in)
	if err != nil {
		t.Errorf("Copy Error: %v", err)
	}
	if int(n) != len(input) {
		t.Errorf("Copy() n=%v, want %v", n, len(input))
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close Error after copied %d bytes: %v", n, err)
	}
	if err := checkCompressedData(out.Bytes(), input); err != nil {
		t.Error(err)
	}
}

func TestEncoderStreams(t *testing.T) {
	// Test that output is streamed.
	// Adjust window size to ensure the encoder outputs at least enough bytes
	// to fill the window.
	const lgWin = 16
	windowSize := int(math.Pow(2, lgWin))
	input := make([]byte, 8*windowSize)
	rand.Read(input)
	out := bytes.Buffer{}
	e := NewWriter(&out, WriterOptions{Quality: 11, LGWin: lgWin})
	halfInput := input[:len(input)/2]
	in := bytes.NewReader(halfInput)

	n, err := io.Copy(e, in)
	if err != nil {
		t.Errorf("Copy Error: %v", err)
	}

	// We've fed more data than the sliding window size. Check that some
	// compressed data has been output.
	if out.Len() == 0 {
		t.Errorf("Output length is 0 after %d bytes written", n)
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close Error after copied %d bytes: %v", n, err)
	}
	if err := checkCompressedData(out.Bytes(), halfInput); err != nil {
		t.Error(err)
	}
}

func TestEncoderLargeInput(t *testing.T) {
	input := make([]byte, 1000000)
	rand.Read(input)
	out := bytes.Buffer{}
	e := NewWriter(&out, WriterOptions{Quality: 5})
	in := bytes.NewReader(input)

	n, err := io.Copy(e, in)
	if err != nil {
		t.Errorf("Copy Error: %v", err)
	}
	if int(n) != len(input) {
		t.Errorf("Copy() n=%v, want %v", n, len(input))
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close Error after copied %d bytes: %v", n, err)
	}
	if err := checkCompressedData(out.Bytes(), input); err != nil {
		t.Error(err)
	}
}

func TestEncoderFlush(t *testing.T) {
	input := make([]byte, 1000)
	rand.Read(input)
	out := bytes.Buffer{}
	e := NewWriter(&out, WriterOptions{Quality: 5})
	in := bytes.NewReader(input)
	_, err := io.Copy(e, in)
	if err != nil {
		t.Fatalf("Copy Error: %v", err)
	}
	if err := e.Flush(); err != nil {
		t.Fatalf("Flush(): %v", err)
	}
	if out.Len() == 0 {
		t.Fatalf("0 bytes written after Flush()")
	}
	decompressed := make([]byte, 1000)
	reader := NewReader(bytes.NewReader(out.Bytes()))
	n, err := reader.Read(decompressed)
	if n != len(decompressed) || err != nil {
		t.Errorf("Expected <%v, nil>, but <%v, %v>", len(decompressed), n, err)
	}
	reader.Close()
	if !bytes.Equal(decompressed, input) {
		t.Errorf(""+
			"Decompress after flush: %v\n"+
			"%q\n"+
			"want:\n%q",
			err, decompressed, input)
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close(): %v", err)
	}
}

type readerWithTimeout struct {
	io.ReadCloser
}

func (r readerWithTimeout) Read(p []byte) (int, error) {
	type result struct {
		n   int
		err error
	}
	ch := make(chan result)
	go func() {
		n, err := r.ReadCloser.Read(p)
		ch <- result{n, err}
	}()
	select {
	case result := <-ch:
		return result.n, result.err
	case <-time.After(5 * time.Second):
		return 0, fmt.Errorf("read timed out")
	}
}

func TestDecoderStreaming(t *testing.T) {
	pr, pw := io.Pipe()
	writer := NewWriter(pw, WriterOptions{Quality: 5, LGWin: 20})
	reader := readerWithTimeout{NewReader(pr)}
	defer func() {
		if err := reader.Close(); err != nil {
			t.Errorf("reader.Close: %v", err)
		}
		go ioutil.ReadAll(pr) // swallow the "EOF" token from writer.Close
		if err := writer.Close(); err != nil {
			t.Errorf("writer.Close: %v", err)
		}
	}()

	ch := make(chan []byte)
	errch := make(chan error)
	go func() {
		for {
			segment, ok := <-ch
			if !ok {
				return
			}
			if n, err := writer.Write(segment); err != nil || n != len(segment) {
				errch <- fmt.Errorf("write=%v,%v, want %v,%v", n, err, len(segment), nil)
				return
			}
			if err := writer.Flush(); err != nil {
				errch <- fmt.Errorf("flush: %v", err)
				return
			}
		}
	}()
	defer close(ch)

	segments := [...][]byte{
		[]byte("first"),
		[]byte("second"),
		[]byte("third"),
	}
	for k, segment := range segments {
		t.Run(fmt.Sprintf("Segment%d", k), func(t *testing.T) {
			select {
			case ch <- segment:
			case err := <-errch:
				t.Fatalf("write: %v", err)
			case <-time.After(5 * time.Second):
				t.Fatalf("timed out")
			}
			wantLen := len(segment)
			got := make([]byte, wantLen)
			if n, err := reader.Read(got); err != nil || n != wantLen || !bytes.Equal(got, segment) {
				t.Fatalf("read[%d]=%q,%v,%v, want %q,%v,%v", k, got, n, err, segment, wantLen, nil)
			}
		})
	}
}

func TestReader(t *testing.T) {
	content := bytes.Repeat([]byte("hello world!"), 10000)
	encoded, _ := Encode(content, WriterOptions{Quality: 5})
	r := NewReader(bytes.NewReader(encoded))
	var decodedOutput bytes.Buffer
	n, err := io.Copy(&decodedOutput, r)
	if err != nil {
		t.Fatalf("Copy(): n=%v, err=%v", n, err)
	}
	if err := r.Close(); err != nil {
		t.Errorf("Close(): %v", err)
	}
	if got := decodedOutput.Bytes(); !bytes.Equal(got, content) {
		t.Errorf(""+
			"Reader output:\n"+
			"%q\n"+
			"want:\n"+
			"<%d bytes>",
			got, len(content))
	}
}

func TestDecode(t *testing.T) {
	content := bytes.Repeat([]byte("hello world!"), 10000)
	encoded, _ := Encode(content, WriterOptions{Quality: 5})
	decoded, err := Decode(encoded)
	if err != nil {
		t.Errorf("Decode: %v", err)
	}
	if !bytes.Equal(decoded, content) {
		t.Errorf(""+
			"Decode content:\n"+
			"%q\n"+
			"want:\n"+
			"<%d bytes>",
			decoded, len(content))
	}
}

func TestDecodeFuzz(t *testing.T) {
	// Test that the decoder terminates with corrupted input.
	content := bytes.Repeat([]byte("hello world!"), 100)
	src := rand.NewSource(0)
	encoded, err := Encode(content, WriterOptions{Quality: 5})
	if err != nil {
		t.Fatalf("Encode(<%d bytes>, _) = _, %s", len(content), err)
	}
	if len(encoded) == 0 {
		t.Fatalf("Encode(<%d bytes>, _) produced empty output", len(content))
	}
	for i := 0; i < 100; i++ {
		enc := append([]byte{}, encoded...)
		for j := 0; j < 5; j++ {
			enc[int(src.Int63())%len(enc)] = byte(src.Int63() % 256)
		}
		Decode(enc)
	}
}

func TestDecodeTrailingData(t *testing.T) {
	content := bytes.Repeat([]byte("hello world!"), 100)
	encoded, _ := Encode(content, WriterOptions{Quality: 5})
	_, err := Decode(append(encoded, 0))
	if err == nil {
		t.Errorf("Expected 'excessive input' error")
	}
}

func TestEncodeDecode(t *testing.T) {
	for _, test := range []struct {
		data    []byte
		repeats int
	}{
		{nil, 0},
		{[]byte("A"), 1},
		{[]byte("<html><body><H1>Hello world</H1></body></html>"), 10},
		{[]byte("<html><body><H1>Hello world</H1></body></html>"), 1000},
	} {
		t.Logf("case %q x %d", test.data, test.repeats)
		input := bytes.Repeat(test.data, test.repeats)
		encoded, err := Encode(input, WriterOptions{Quality: 5})
		if err != nil {
			t.Errorf("Encode: %v", err)
		}
		// Inputs are compressible, but may be too small to compress.
		if maxSize := len(input)/2 + 20; len(encoded) >= maxSize {
			t.Errorf(""+
				"Encode returned %d bytes, want <%d\n"+
				"Encoded=%q",
				len(encoded), maxSize, encoded)
		}
		decoded, err := Decode(encoded)
		if err != nil {
			t.Errorf("Decode: %v", err)
		}
		if !bytes.Equal(decoded, input) {
			var want string
			if len(input) > 320 {
				want = fmt.Sprintf("<%d bytes>", len(input))
			} else {
				want = fmt.Sprintf("%q", input)
			}
			t.Errorf(""+
				"Decode content:\n"+
				"%q\n"+
				"want:\n"+
				"%s",
				decoded, want)
		}
	}
}
