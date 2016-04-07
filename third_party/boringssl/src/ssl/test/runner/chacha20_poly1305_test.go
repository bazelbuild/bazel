package runner

import (
	"bytes"
	"encoding/hex"
	"testing"
)

// See draft-irtf-cfrg-chacha20-poly1305-10, section 2.1.1.
func TestChaChaQuarterRound(t *testing.T) {
	state := [16]uint32{0x11111111, 0x01020304, 0x9b8d6f43, 0x01234567}
	chaChaQuarterRound(&state, 0, 1, 2, 3)

	a, b, c, d := state[0], state[1], state[2], state[3]
	if a != 0xea2a92f4 || b != 0xcb1cf8ce || c != 0x4581472e || d != 0x5881c4bb {
		t.Errorf("Incorrect results: %x", state)
	}
}

// See draft-irtf-cfrg-chacha20-poly1305-10, section 2.2.1.
func TestChaChaQuarterRoundState(t *testing.T) {
	state := [16]uint32{
		0x879531e0, 0xc5ecf37d, 0x516461b1, 0xc9a62f8a,
		0x44c20ef3, 0x3390af7f, 0xd9fc690b, 0x2a5f714c,
		0x53372767, 0xb00a5631, 0x974c541a, 0x359e9963,
		0x5c971061, 0x3d631689, 0x2098d9d6, 0x91dbd320,
	}
	chaChaQuarterRound(&state, 2, 7, 8, 13)

	expected := [16]uint32{
		0x879531e0, 0xc5ecf37d, 0xbdb886dc, 0xc9a62f8a,
		0x44c20ef3, 0x3390af7f, 0xd9fc690b, 0xcfacafd2,
		0xe46bea80, 0xb00a5631, 0x974c541a, 0x359e9963,
		0x5c971061, 0xccc07c79, 0x2098d9d6, 0x91dbd320,
	}
	for i := range state {
		if state[i] != expected[i] {
			t.Errorf("Mismatch at %d: %x vs %x", i, state, expected)
		}
	}
}

// See draft-irtf-cfrg-chacha20-poly1305-10, section 2.3.2.
func TestChaCha20Block(t *testing.T) {
	state := [16]uint32{
		0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
		0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
		0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
		0x00000001, 0x09000000, 0x4a000000, 0x00000000,
	}
	out := make([]byte, 64)
	chaCha20Block(&state, out)

	expected := []byte{
		0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15,
		0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4,
		0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03,
		0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e,
		0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09,
		0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2,
		0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
		0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
	}
	if !bytes.Equal(out, expected) {
		t.Errorf("Got %x, wanted %x", out, expected)
	}
}

// See draft-agl-tls-chacha20poly1305-04, section 7.
func TestChaCha20Poly1305(t *testing.T) {
	key, _ := hex.DecodeString("4290bcb154173531f314af57f3be3b5006da371ece272afa1b5dbdd1100a1007")
	input, _ := hex.DecodeString("86d09974840bded2a5ca")
	nonce, _ := hex.DecodeString("cd7cf67be39c794a")
	ad, _ := hex.DecodeString("87e229d4500845a079c0")
	output, _ := hex.DecodeString("e3e446f7ede9a19b62a4677dabf4e3d24b876bb284753896e1d6")

	aead, err := newChaCha20Poly1305(key)
	if err != nil {
		t.Fatal(err)
	}

	out, err := aead.Open(nil, nonce, output, ad)
	if err != nil {
		t.Errorf("Open failed: %s", err)
	} else if !bytes.Equal(out, input) {
		t.Errorf("Open gave %x, wanted %x", out, input)
	}

	out = aead.Seal(nil, nonce, input, ad)
	if !bytes.Equal(out, output) {
		t.Errorf("Open gave %x, wanted %x", out, output)
	}

	out[0]++
	_, err = aead.Open(nil, nonce, out, ad)
	if err == nil {
		t.Errorf("Open on malformed data unexpectedly succeeded")
	}
}
