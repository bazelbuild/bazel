package runner

import (
	"crypto/cipher"
	"crypto/subtle"
	"encoding/binary"
	"errors"
)

// See draft-agl-tls-chacha20poly1305-04 and
// draft-irtf-cfrg-chacha20-poly1305-10. Where the two differ, the
// draft-agl-tls-chacha20poly1305-04 variant is implemented.

func leftRotate(a uint32, n uint) uint32 {
	return (a << n) | (a >> (32 - n))
}

func chaChaQuarterRound(state *[16]uint32, a, b, c, d int) {
	state[a] += state[b]
	state[d] = leftRotate(state[d]^state[a], 16)

	state[c] += state[d]
	state[b] = leftRotate(state[b]^state[c], 12)

	state[a] += state[b]
	state[d] = leftRotate(state[d]^state[a], 8)

	state[c] += state[d]
	state[b] = leftRotate(state[b]^state[c], 7)
}

func chaCha20Block(state *[16]uint32, out []byte) {
	var workingState [16]uint32
	copy(workingState[:], state[:])
	for i := 0; i < 10; i++ {
		chaChaQuarterRound(&workingState, 0, 4, 8, 12)
		chaChaQuarterRound(&workingState, 1, 5, 9, 13)
		chaChaQuarterRound(&workingState, 2, 6, 10, 14)
		chaChaQuarterRound(&workingState, 3, 7, 11, 15)
		chaChaQuarterRound(&workingState, 0, 5, 10, 15)
		chaChaQuarterRound(&workingState, 1, 6, 11, 12)
		chaChaQuarterRound(&workingState, 2, 7, 8, 13)
		chaChaQuarterRound(&workingState, 3, 4, 9, 14)
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], workingState[i]+state[i])
	}
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

type chaCha20Poly1305 struct {
	key [32]byte
}

func newChaCha20Poly1305(key []byte) (cipher.AEAD, error) {
	if len(key) != 32 {
		return nil, errors.New("bad key length")
	}
	aead := new(chaCha20Poly1305)
	copy(aead.key[:], key)
	return aead, nil
}

func (c *chaCha20Poly1305) NonceSize() int { return 8 }
func (c *chaCha20Poly1305) Overhead() int  { return 16 }

func (c *chaCha20Poly1305) chaCha20(out, in, nonce []byte, counter uint64) {
	var state [16]uint32
	state[0] = 0x61707865
	state[1] = 0x3320646e
	state[2] = 0x79622d32
	state[3] = 0x6b206574
	for i := 0; i < 8; i++ {
		state[4+i] = binary.LittleEndian.Uint32(c.key[i*4 : i*4+4])
	}
	state[14] = binary.LittleEndian.Uint32(nonce[0:4])
	state[15] = binary.LittleEndian.Uint32(nonce[4:8])

	for i := 0; i < len(in); i += 64 {
		state[12] = uint32(counter & 0xffffffff)
		state[13] = uint32(counter >> 32)

		var tmp [64]byte
		chaCha20Block(&state, tmp[:])
		count := 64
		if len(in)-i < count {
			count = len(in) - i
		}
		for j := 0; j < count; j++ {
			out[i+j] = in[i+j] ^ tmp[j]
		}

		counter++
	}
}

func (c *chaCha20Poly1305) poly1305(tag *[16]byte, nonce, ciphertext, additionalData []byte) {
	input := make([]byte, 0, len(additionalData)+8+len(ciphertext)+8)
	input = append(input, additionalData...)
	input, out := sliceForAppend(input, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(additionalData)))
	input = append(input, ciphertext...)
	input, out = sliceForAppend(input, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(ciphertext)))

	var poly1305Key [32]byte
	c.chaCha20(poly1305Key[:], poly1305Key[:], nonce, 0)

	poly1305Sum(tag, input, &poly1305Key)
}

func (c *chaCha20Poly1305) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if len(nonce) != 8 {
		panic("Bad nonce length")
	}

	ret, out := sliceForAppend(dst, len(plaintext)+16)
	c.chaCha20(out[:len(plaintext)], plaintext, nonce, 1)

	var tag [16]byte
	c.poly1305(&tag, nonce, out[:len(plaintext)], additionalData)
	copy(out[len(plaintext):], tag[:])

	return ret
}

func (c *chaCha20Poly1305) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(nonce) != 8 {
		panic("Bad nonce length")
	}
	if len(ciphertext) < 16 {
		return nil, errors.New("chacha20: message authentication failed")
	}
	plaintextLen := len(ciphertext) - 16

	var tag [16]byte
	c.poly1305(&tag, nonce, ciphertext[:plaintextLen], additionalData)
	if subtle.ConstantTimeCompare(tag[:], ciphertext[plaintextLen:]) != 1 {
		return nil, errors.New("chacha20: message authentication failed")
	}

	ret, out := sliceForAppend(dst, plaintextLen)
	c.chaCha20(out, ciphertext[:plaintextLen], nonce, 1)
	return ret, nil
}
