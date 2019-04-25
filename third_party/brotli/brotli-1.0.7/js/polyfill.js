if (!Int32Array.__proto__.from) {
  Object.defineProperty(Int32Array.__proto__, 'from', {
    value: function(obj) {
      obj = Object(obj);
      if (!obj['length']) {
        return new this(0);
      }
      var typed_array = new this(obj.length);
      for(var i = 0; i < typed_array.length; i++) {
        typed_array[i] = obj[i];
      }
      return typed_array;
    }
  });
}

if (!Array.prototype.copyWithin) {
  Array.prototype.copyWithin = function(target, start, end) {
    var O = Object(this);
    var len = O.length >>> 0;
    var to = target | 0;
    var from = start | 0;
    var count = Math.min(Math.min(end | 0, len) - from, len - to);
    var direction = 1;
    if (from < to && to < (from + count)) {
      direction = -1;
      from += count - 1;
      to += count - 1;
    }
    while (count > 0) {
      O[to] = O[from];
      from += direction;
      to += direction;
      count--;
    }
    return O;
  };
}

if (!Array.prototype.fill) {
  Object.defineProperty(Array.prototype, 'fill', {
    value: function(value, start, end) {
      end = end | 0;
      var O = Object(this);
      var k = start | 0;
      while (k < end) {
        O[k] = value;
        k++;
      }
      return O;
    }
  });
}

if (!Int8Array.prototype.copyWithin) {
  Int8Array.prototype.copyWithin = Array.prototype.copyWithin;
}

if (!Int8Array.prototype.fill) {
  Int8Array.prototype.fill = Array.prototype.fill;
}

if (!Int32Array.prototype.fill) {
  Int32Array.prototype.fill = Array.prototype.fill;
}
