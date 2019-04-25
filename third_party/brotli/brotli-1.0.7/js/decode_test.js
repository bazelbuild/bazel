goog.require('goog.testing.asserts');
goog.require('goog.testing.jsunit');

/**
 * @param {!Int8Array} bytes
 * @return {string}
 */
function bytesToString(bytes) {
  return String.fromCharCode.apply(null, new Uint16Array(bytes));
}

function testMetadata() {
  assertEquals("", bytesToString(BrotliDecode(Int8Array.from([1, 11, 0, 42, 3]))));
}

function testEmpty() {
  assertEquals("", bytesToString(BrotliDecode(Int8Array.from([6]))));
  assertEquals("", bytesToString(BrotliDecode(Int8Array.from([0x81, 1]))));
}

function testBaseDictWord() {
  var input = Int8Array.from([
    0x1b, 0x03, 0x00, 0x00, 0x00, 0x00, 0x80, 0xe3, 0xb4, 0x0d, 0x00, 0x00,
    0x07, 0x5b, 0x26, 0x31, 0x40, 0x02, 0x00, 0xe0, 0x4e, 0x1b, 0x41, 0x02
  ]);
  /** @type {!Int8Array} */
  var output = BrotliDecode(input);
  assertEquals("time", bytesToString(output));
}

function testBlockCountMessage() {
  var input = Int8Array.from([
    0x1b, 0x0b, 0x00, 0x11, 0x01, 0x8c, 0xc1, 0xc5, 0x0d, 0x08, 0x00, 0x22,
    0x65, 0xe1, 0xfc, 0xfd, 0x22, 0x2c, 0xc4, 0x00, 0x00, 0x38, 0xd8, 0x32,
    0x89, 0x01, 0x12, 0x00, 0x00, 0x77, 0xda, 0x04, 0x10, 0x42, 0x00, 0x00, 0x00
  ]);
  /** @type {!Int8Array} */
  var output = BrotliDecode(input);
  assertEquals("aabbaaaaabab", bytesToString(output));
}

function testCompressedUncompressedShortCompressedSmallWindow() {
  var input = Int8Array.from([
    0x21, 0xf4, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x1c, 0xa7, 0x6d, 0x00, 0x00,
    0x38, 0xd8, 0x32, 0x89, 0x01, 0x12, 0x00, 0x00, 0x77, 0xda, 0x34, 0x7b,
    0xdb, 0x50, 0x80, 0x02, 0x80, 0x62, 0x62, 0x62, 0x62, 0x62, 0x62, 0x31,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x4e, 0xdb, 0x00, 0x00, 0x70, 0xb0,
    0x65, 0x12, 0x03, 0x24, 0x00, 0x00, 0xee, 0xb4, 0x11, 0x24, 0x00
  ]);
  /** @type {!Int8Array} */
  var output = BrotliDecode(input);
  assertEquals(
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
    "aaaaaaaaaaaaaabbbbbbbbbb", bytesToString(output));
}

function testIntactDistanceRingBuffer0() {
  var input = Int8Array.from([
    0x1b, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x80, 0xe3, 0xb4, 0x0d, 0x00, 0x00,
    0x07, 0x5b, 0x26, 0x31, 0x40, 0x02, 0x00, 0xe0, 0x4e, 0x1b, 0xa1, 0x80,
    0x20, 0x00
  ]);
  /** @type {!Int8Array} */
  var output = BrotliDecode(input);
  assertEquals("himselfself", bytesToString(output));
}
