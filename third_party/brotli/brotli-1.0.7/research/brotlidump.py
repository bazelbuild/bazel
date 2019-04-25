#!python3
"""Program to dump contents of Brotli compressed files showing the compression format.
Jurjen N.E. Bos, 2016.
I found the following issues with the Brotli format:
- The distance alphabet has size 16+(48<<POSTFIX),
  but the last symbols are useless.
  It could be lowered to 16+(44-POSTFIX<<POSTFIX), and this could matter.
- The block type code is useless if NBLTYPES==2, you would only need 1 symbol
  anyway, so why don't you just switch to "the other" type?
"""
import struct
from operator import itemgetter, methodcaller
from itertools import accumulate, repeat
from collections import defaultdict, deque
from functools import partial

class InvalidStream(Exception): pass
#lookup table
L, I, D = "literal", "insert&copy", "distance"
pL, pI, pD = 'P'+L, 'P'+I, 'P'+D

def outputCharFormatter(c):
    """Show character in readable format
    """
    #TODO 2: allow hex only output
    if 32<c<127: return chr(c)
    elif c==10: return '\\n'
    elif c==13: return '\\r'
    elif c==32: return '" "'
    else: return '\\x{:02x}'.format(c)

def outputFormatter(s):
    """Show string or char.
    """
    result = ''
    def formatSubString(s):
        for c in s:
            if c==32: yield ' '
            else: yield outputCharFormatter(c)
    if len(result)<200: return ''.join(formatSubString(s))
    else:
        return ''.join(formatSubString(s[:100]))+'...'+ \
               ''.join(formatSubString(s[-100:]))


class BitStream:
    """Represent a bytes object. Can read bits and prefix codes the way
    Brotli does.
    """
    def __init__(self, byteString):
        self.data = byteString
        #position in bits: byte pos is pos>>3, bit pos is pos&7
        self.pos = 0

    def __repr__(self):
        """Representation
        >>> olleke
        BitStream(pos=0:0)
        """
        return "BitStream(pos={:x}:{})".format(self.pos>>3, self.pos&7)

    def read(self, n):
        """Read n bits from the stream and return as an integer.
        Produces zero bits beyond the stream.
        >>> olleke.data[0]==27
        True
        >>> olleke.read(5)
        27

        >>> olleke
        BitStream(pos=0:5)
        """
        value = self.peek(n)
        self.pos += n
        if self.pos>len(self.data)*8:
            raise ValueError('Read past end of stream')
        return value

    def peek(self, n):
        """Peek an n bit integer from the stream without updating the pointer.
        It is not an error to read beyond the end of the stream.
        >>> olleke.data[:2]==b'\x1b\x2e' and 0x2e1b==11803
        True
        >>> olleke.peek(15)
        11803
        >>> hex(olleke.peek(32))
        '0x2e1b'
        """
        #read bytes that contain the data: self.data[self.pos>>3:self.pos+n+7>>3]
        #convert to int: int.from_bytes(..., 'little')
        #shift out the bits from the first byte: >>(self.pos&7)
        #mask unwanted bits: & (1<<n)-1
        return int.from_bytes(
            self.data[self.pos>>3:self.pos+n+7>>3],
            'little')>>(self.pos&7) & (1<<n)-1

    def readBytes(self, n):
        """Read n bytes from the stream on a byte boundary.
        """
        if self.pos&7: raise ValueError('readBytes: need byte boundary')
        result = self.data[self.pos>>3:(self.pos>>3)+n]
        self.pos += 8*n
        return result

#-----------------------Symbol-------------------------------------------
class Symbol:
    """A symbol in a code.
    Refers back to the code that contains it.
    Index is the place in the alphabet of the symbol.
    """
    __slots__ = 'code', 'index'
    def __init__(self, code, value):
        self.code = code
        self.index = value

    def __repr__(self):
        return 'Symbol({}, {})'.format(self.code.name, self.index)

    def __len__(self):
        """Number of bits in the prefix notation of this symbol
        """
        return self.code.length(self.index)

    def __int__(self):
        return self.index

    #these routines call equivalent routine in Code class
    def bitPattern(self):
        """Value of the symbol in the stream
        """
        return self.code.bitPattern(self.index)

    def extraBits(self):
        """Number of extra bits to read for this symbol
        """
        return self.code.extraBits(self.index)

    def __str__(self):
        """Short descriptor of the symbol without extra bits.
        """
        return self.code.mnemonic(self.index)

    #requiring optional extra bits, if self.code supports them
    def value(self, extra=None):
        """The value used for processing. Can be a tuple.
        with optional extra bits
        """
        if isinstance(self.code, WithExtra):
            if not 0<=extra<1<<self.extraBits():
                raise ValueError("value: extra value doesn't fit in extraBits")
            return self.code.value(self.index, extra)
        if extra is not None:
            raise ValueError('value: no extra bits for this code')
        return self.code.value(self.index)

    def explanation(self, extra=None):
        """Long explanation of the value from the numeric value
        with optional extra bits
        Used by Layout.verboseRead when printing the value
        """
        if isinstance(self.code, WithExtra):
            return self.code.callback(self, extra)
        return self.code.callback(self)

#========================Code definitions==================================
class RangeDecoder:
    """A decoder for the Code class that assumes the symbols
    are encoded consecutively in binary.
    It all depends on the "alphabetSize" property.
    The range runs from 0 to alphabetSize-1.
    This is the default decoder.
    """
    def __init__(self, *, alphabetSize=None, bitLength=None, **args):
        if bitLength is not None: alphabetSize = 1<<bitLength
        if alphabetSize is not None:
            self.alphabetSize = alphabetSize
            self.maxLength = (alphabetSize-1).bit_length()

    def __len__(self):
        return self.alphabetSize

    def __iter__(self):
        """Produce all symbols.
        """
        return map(partial(Symbol, self), range(len(self)))

    def __getitem__(self, index):
        if index>=self.alphabetSize: raise ValueError('index out of range')
        return Symbol(self, index)

    def bitPattern(self, index):
        return '{:0{}b}'.format(index, self.maxLength)

    def length(self, index):
        """Encoding length of given symbol.
        Does not depend on index in this case.
        """
        return self.maxLength

    def decodePeek(self, data):
        """Find which symbol index matches the given data (from peek, as a number)
        and return the number of bits decoded.
        Can also be used to figure out length of a symbol.
        """
        return self.maxLength, Symbol(self, data&(1<<self.maxLength)-1)

class PrefixDecoder:
    """A decoder for the Code class that uses a prefix code.
    The code is determined by encoding:
    encode[p] gives the index corresponding to bit pattern p.
    Used setDecode(decodeTable) to switch the decoder from the default
    to a prefix decoder, or pass decodeTable at init.
    You can also use setLength(lengthTable)
    to define the encoding from the lengths.
    The set of symbol values does not need to be consecutive.
    """
    def __init__(self, *, decodeTable=None, **args):
        if decodeTable is not None: self.setDecode(decodeTable)

    def __len__(self):
        return len(self.decodeTable)

    def __iter__(self):
        def revBits(index):
            return self.bitPattern(index)[::-1]
        return (
            Symbol(self, index)
            for index in sorted(self.decodeTable.values(), key=revBits)
            )

    def __getitem__(self, index):
        if index not in self.lengthTable:
            raise ValueError('No symbol {}[{}]'.format(
                self.__class__.__name__, index))
        return Symbol(self, index)

    def bitPattern(self, index):
        bits = next(b for (b,s) in self.decodeTable.items() if s==index)
        return '{:0{}b}'.format(bits, self.length(index))

    def length(self, index):
        """Encoding length of given symbol.
        """
        return self.lengthTable[index]

    def decodePeek(self, data):
        """Find which symbol index matches the given data (from peek, as a number)
        and return the number of bits decoded.
        Can also be used to figure out length of a symbol.
        """
        #do binary search for word length
        #invariant: lo<=length<=hi
        lo, hi = self.minLength, self.maxLength
        while lo<=hi:
            mid = lo+hi>>1
            #note lo<=mid<hi at this point
            mask = (1<<mid)-1
            #lets see what happens if we guess length is mid
            try: index = self.decodeTable[data&mask]
            except KeyError:
                #too many bits specified, reduce estimated length
                hi = mid-1
                continue
            #we found a symbol, but there could be a longer match
            symbolLength = self.lengthTable[index]
            if symbolLength<=mid:
                #all bits match, symbol must be right
                return symbolLength, Symbol(self, index)
            #there must be more bits to match
            lo = mid+1
        return lo, Symbol(self, index)

    #routine to set up the tables
    def setDecode(self, decodeTable):
        """Store decodeTable,
        and compute lengthTable, minLength, maxLength from encodings.
        """
        self.decodeTable = decodeTable
        #set of symbols with unknown length
        todo = set(decodeTable)
        #bit size under investigation
        maskLength = 0
        lengthTable = {}
        while todo:
            mask = (1<<maskLength)-1
            #split the encodings that we didn't find yet using b bits
            splitSymbols = defaultdict(list)
            for s in todo: splitSymbols[s&mask].append(s)
            #unique encodings have a length of maskLength bits
            #set length, and remove from todo list
            for s,subset in splitSymbols.items():
                if len(subset)==1:
                    lengthTable[self.decodeTable[s]] = maskLength
                    todo.remove(s)
            #now investigate with longer mask
            maskLength +=1
        #save result
        self.lengthTable = lengthTable
        self.minLength = min(lengthTable.values())
        self.maxLength = max(lengthTable.values())
        self.switchToPrefix()

    def setLength(self, lengthTable):
        """Given the bit pattern lengths for symbols given in lengthTable,
        set decodeTable, minLength, maxLength
        """
        self.lengthTable = lengthTable
        self.minLength = min(lengthTable.values())
        self.maxLength = max(lengthTable.values())
        #compute the backwards codes first; then reverse them
        #compute (backwards) first code for every separate lengths
        nextCodes = []
        #build codes for each length, from right to left
        code = 0
        for bits in range(self.maxLength+1):
            code <<= 1
            nextCodes.append(code)
            code += sum(x==bits for x in lengthTable.values())
        self.decodeTable = {}
        #count codes for each length, and store reversed in the table
        for symbol in sorted(lengthTable):
            bits = lengthTable[symbol]
            bitpattern = '{:0{}b}'.format(nextCodes[bits], bits)
            self.decodeTable[int(bitpattern[::-1], 2)] = symbol
            nextCodes[bits] += 1
        self.switchToPrefix()

    def switchToPrefix(self):
        """This routine makes sure the prefix decoder is activated.
        """
        self.mode = PrefixDecoder

class Code(RangeDecoder, PrefixDecoder):
    """An alphabet of symbols, that can be read from a stream.
    If you use setDecode or setLength, you have a prefix code,
    otherwise you have a range code.
    Features:
    code[index] produces symbol with given index
    value(index): value of symbol
    mnemonic(index): short description of symbol
    explanation(index): show meaning of symbol, shown in Layout.verboseRead
    iter(code): produce all symbols in some order
    name: show as context in Layout.verboseRead
    """
    name = '?'
    #callback is a function that gets the symbol and the extra bits
    #default callback calls explanation
    def __init__(self, name=None, *, callback=None, description='', **args):
        """Don't forget to set either alphabetSize or decodeTable
        """
        #set name when provided, otherwise take class variable
        if name is not None: self.name = name
        if callback is not None: self.callback = callback
        self.description = description
        #mode switch
        if 'bitLength' in args or 'alphabetSize' in args:
            self.mode = RangeDecoder
            RangeDecoder.__init__(self, **args)
        elif 'decodeTable' in args:
            self.mode = PrefixDecoder
            PrefixDecoder.__init__(self, **args)
        else:
            super().__init__(**args)

    def __repr__(self):
        return self.__class__.__name__+' '+self.name

    #the routines that get switched between RangeDecoder and PrefixDecoder
    def __len__(self): return self.mode.__len__(self)
    def __iter__(self): return self.mode.__iter__(self)
    def __getitem__(self, index): return self.mode.__getitem__(self, index)
    def bitPattern(self, index): return self.mode.bitPattern(self, index)
    def length(self, index): return self.mode.length(self, index)
    def decodePeek(self, data): return self.mode.decodePeek(self, data)
    #general routines
    def value(self, index, extra=None):
        """Get value of symbol for computations.
        Override where needed.
        """
        if extra is not None:
            raise ValueError('value: no extra for this symbol')
        return index

    def mnemonic(self, index):
        """Give mnemonic of symbol.
        Override where needed.
        """
        return str(self.value(index))

    def callback(self, symbol):
        return self.explanation(symbol.index)

    def explanation(self, index):
        """Long explanation of the value from the numeric value
        This is a default routine.
        You can customize in three ways:
        - set description to add some text
        - override to get more control
        - set callback to make it dependent on you local variables
        """
        value = self.value(index)
        return '{0}{1}: {2}'.format(
            self.description and self.description+': ',
            self.bitPattern(index),
            value,
            )

    def extraBits(self, index):
        return 0

    #Routines that use the decode interface
    def showCode(self, width=80):
        """Show all words of the code in a nice format.
        """
        #make table of all symbols with binary strings
        symbolStrings = [
            (self.bitPattern(s.index), self.mnemonic(s.index))
            for s in self
            ]
        #determine column widths the way Lisp programmers do it
        leftColWidth, rightColWidth = map(max, map(
            map,
            repeat(len),
            zip(*symbolStrings)
            ))
        colwidth = leftColWidth+rightColWidth
        columns = 81//(colwidth+2)
        rows = -(-len(symbolStrings)//columns)
        def justify(bs):
            b,s = bs
            return b.rjust(leftColWidth)+':'+s.ljust(rightColWidth)
        for i in range(rows):
            print(' '.join(map(justify, symbolStrings[i::rows])).rstrip())

    def readTuple(self, stream):
        """Read symbol from stream. Returns symbol, length.
        """
        length, symbol = self.decodePeek(stream.peek(self.maxLength))
        stream.pos += length
        return length, symbol

    def readTupleAndExtra(self, stream):
        return self.readTuple(stream)+(0, None)

class WithExtra(Code):
    """Extension for Code so that symbol may have extra bits associated.
    If you supply an extraTable, you can use extraBits
    You can define an extraTable,
    which allows to call extraBits to get the number of extraBits.
    Otherwise, you can supply extraBits yourself.
    Routine readTupleAndExtra now reads the extra bits too.
    Value probably needs to be overridden; see Enumerator.
    Note: this does not give you an decodeTable.
    """
    #redefine these if you don't want to use an extraTable
    def extraBits(self, index):
        """Get the number of extra bits for this symbol.
        """
        return self.extraTable[index]

    def mnemonic(self, index):
        """This value must be independent of extra.
        """
        return str(index)

    def readTupleAndExtra(self, stream):
        """Read symbol and extrabits from stream.
        Returns symbol length, symbol, extraBits, extra
        >>> olleke.pos = 6
        >>> MetablockLengthAlphabet().readTupleAndExtra(olleke)
        (2, Symbol(MLEN, 4), 16, 46)
        """
        length, symbol = self.decodePeek(stream.peek(self.maxLength))
        stream.pos += length
        extraBits = self.extraBits(symbol.index)
        return length, symbol, extraBits, stream.read(extraBits)

    def explanation(self, index, extra=None):
        """Expanded version of Code.explanation supporting extra bits.
        If you don't supply extra, it is not mentioned.
        """
        extraBits = 0 if extra is None else self.extraBits(index)
        if not hasattr(self, 'extraTable'):
            formatString = '{0}{3}'
            lo = hi = value = self.value(index, extra)
        elif extraBits==0:
            formatString = '{0}{2}: {3}'
            lo, hi = self.span(index)
            value = lo
        else:
            formatString = '{0}{1} {2}: {3}-{4}; {3}+{5}={6}'
            lo, hi = self.span(index)
            value = lo+extra
        return formatString.format(
            self.description and self.description+': ',
            'x'*extraBits,
            self.bitPattern(index),
            lo, hi,
            extra,
            value,
            )

    def callback(self, symbol, extra):
        return self.explanation(symbol.index, extra)

class BoolCode(Code):
    """Same as Code(bitLength=1), but shows a boolean.
    """
    def __init__(self, name=None, **args):
        super().__init__(name, bitLength=1, **args)

    def value(self, index, extra=None):
        return bool(super().value(index, extra))

class Enumerator(WithExtra):
    """Code that is defined by the ExtraTable.
    extraTable is a class variable that contains
    the extraBits of the symbols from 0
    value0 contains the value of symbol 0
    encodings is not neccessary, but allowed.
    Note: place for FixedCode to make sure extraBits works
    """
    def __init__(self, name=None, **args):
        #if there is no decodeTable to determine length, compute it ourselves
        if 'decodeTable' not in args:
            args['alphabetSize'] = len(self.extraTable)
        super().__init__(name, **args)

    def __len__(self):
        return len(self.extraTable)

    def __getitem__(self, index):
        """Faster than PrefixDecoder
        """
        if index>=len(self.extraTable):
            raise ValueError("No symbol {}[{}]".format(
                self.__class__.__name__, index))
        return Symbol(self, index)

    def value(self, index, extra):
        """Override if you don't define value0 and extraTable
        """
        lower, upper = self.span(index)
        value = lower+(extra or 0)
        if value>upper:
            raise ValueError('value: extra out of range')
        return value

    def span(self, index):
        """Give the range of possible values in a tuple
        Useful for mnemonic and explanation
        """
        lower = self.value0+sum(1<<x for x in self.extraTable[:index])
        upper = lower+(1<<self.extraTable[index])
        return lower, upper-1

#======================Code subclasses======================================
#Alphabets used in the metablock header----------------------------------
#For prefix codes
class PrefixCodeHeader(WithExtra):
    """Header of prefix codes.
    """
    def __init__(self, codename):
        super().__init__('PFX', bitLength=2)
        #this is the name of the code that it describes
        self.codename = codename

    def extraBits(self, index):
        return 2 if index==1 else 0

    def value(self, index, extra):
        """Returns ('Simple', #codewords) or ('Complex', HSKIP)
        """
        if index==1:
            if extra>3:
                raise ValueError('value: extra out of range')
            return 'Simple', extra+1
        if extra:
            raise ValueError('value: extra out of range')
        return 'Complex', index

    def explanation(self, index, extra):
        if index==1:
            return '{} is simple with {} code word{}'.format(
                self.codename, extra+1, 's' if extra else '')
        lengths = [1, 2, 3, 4, 0, 5, 17, 6]
        return '{} is complex with lengths {}...'.format(
            self.codename,
            ','.join(
                map(str, lengths[index:index+5]))
            )

class TreeShapeAlhabet(BoolCode):
    """The bit used to indicate if four word code is "deep" or "wide"
    """
    name = 'SHAPE'
    def value(self, index):
        return [(2,2,2,2), (1,2,3,3)][index]

    def explanation(self, index):
        return str(bool(index))+': lengths {},{},{},{}'.format(*self.value(index))

class LengthOfLengthAlphabet(Code):
    """For use in decoding complex code descriptors.
    >>> lengthOfLengthAlphabet = LengthOfLengthAlphabet('')
    >>> print(lengthOfLengthAlphabet[2])
    coded with 2 bits
    >>> len(lengthOfLengthAlphabet[0])
    2
    >>> [len(lengthOfLengthAlphabet[x]) for x in range(6)]
    [2, 4, 3, 2, 2, 4]
    >>> lengthOfLengthAlphabet.showCode()
      00:skipped             01:coded with 4 bits 0111:coded with 1 bits
      10:coded with 3 bits  011:coded with 2 bits 1111:coded with 5 bits
    """
    decodeTable = {
         0b00:0,     0b10:3,
       0b0111:1,     0b01:4,
        0b011:2,   0b1111:5,
       }

    def __init__(self, name=None, **args):
        super().__init__(name, decodeTable=self.decodeTable, **args)

    def mnemonic(self, index):
        if index==0: return 'skipped'
        return 'coded with {} bits'.format(index)

    def explanation(self, index, extra=None):
        return self.description+': '+self.mnemonic(index)

class LengthAlphabet(WithExtra):
    """Length of symbols
    Used during construction of a code.
    """
    def __init__(self, name):
        super().__init__(name, alphabetSize=18)

    def extraBits(self, index):
        return {16:2, 17:3}.get(index, 0)

    def mnemonic(self, index):
        if index==0: return 'unused'
        elif index==16: return 'rep xx'
        elif index==17: return 'zero xxx'
        else: return 'len {}'.format(index)

    def explanation(self, index, extra):
        return self.description.format(self[index], extra)

    def value(self, index, extra):
        #the caller got the length already, so extra is enough
        return extra

#Stream header
class WindowSizeAlphabet(Code):
    """The alphabet used for window size in the stream header.
    >>> WindowSizeAlphabet()[10].explanation()
    'windowsize=(1<<10)-16=1008'
    """
    decodeTable = {
        0b0100001: 10,   0b1100001: 14,   0b0011: 18,   0b1011: 22,
        0b0110001: 11,   0b1110001: 15,   0b0101: 19,   0b1101: 23,
        0b1000001: 12,         0b0: 16,   0b0111: 20,   0b1111: 24,
        0b1010001: 13,   0b0000001: 17,   0b1001: 21,
        0b0010001: None,
        }

    name = 'WSIZE'

    def __init__(self, name=None):
        super().__init__(name, decodeTable=self.decodeTable)

    def value(self, index):
        #missing value gives index None
        if index is None: return None
        return (1<<index)-16

    def explanation(self, index):
        return 'windowsize=(1<<{})-16={}'.format(
            index, (1<<index)-16)

#Metablock
class MetablockLengthAlphabet(WithExtra):
    """Used for the meta block length;
    also indicates a block with no data
    >>> metablockLengthAlphabet = MetablockLengthAlphabet()
    >>> metablockLengthAlphabet[0]; str(metablockLengthAlphabet[0])
    Symbol(MLEN, 0)
    'empty'
    >>> metablockLengthAlphabet[3]
    Traceback (most recent call last):
        ...
    ValueError: No symbol MetablockLengthAlphabet[3]
    >>> print(metablockLengthAlphabet[4])
    hhhh00
    >>> metablockLengthAlphabet[4].value(0x1000)
    4097
    >>> metablockLengthAlphabet[5].value(0x1000)
    Traceback (most recent call last):
        ...
    InvalidStream: Zeros in high nibble of MLEN
    >>> metablockLengthAlphabet[5].explanation(0x12345)
    'data length: 12345h+1=74566'
    >>> metablockLengthAlphabet.showCode()
    00:hhhh00   10:hhhhhh10 01:hhhhh01  11:empty
    """
    decodeTable = {0b11:0, 0b00:4, 0b01:5, 0b10:6}

    name = 'MLEN'
    def __init__(self, name=None):
        super().__init__(name, decodeTable=self.decodeTable)

    def extraBits(self, index):
        return index*4

    def mnemonic(self, index):
        if index==0: return 'empty'
        return 'h'*(self.extraBits(index)//4)+self.bitPattern(index)

    def value(self, index, extra):
        extraBits = self.extraBits(index)
        if not 0<=extra<1<<extraBits:
            raise ValueError('value: extra out of range')
        if index==0: return 0
        if index>4 and extra>>extraBits-4==0: raise InvalidStream(
            'Zeros in high nibble of MLEN')
        return extra+1

    def explanation(self, index, extra):
        if index==0: return '11: empty block'
        extraBits = self.extraBits(index)
        return 'data length: {:0{}x}h+1={}'.format(extra, extraBits//4, extra+1)


class ReservedAlphabet(BoolCode):
    """The reserved bit that must be zero.
    """
    name = 'RSVD'
    def value(self, index):
        if index: raise ValueError('Reserved bit is not zero')

    def explanation(self, index):
        return 'Reserved (must be zero)'

class FillerAlphabet(Code):
    def __init__(self, *, streamPos):
        super().__init__('SKIP', bitLength=(-streamPos)&7)

    def explanation(self, index):
        return '{} bit{} ignored'.format(
            self.length(index),
            '' if self.length(index)==1 else 's',
            )

class SkipLengthAlphabet(WithExtra):
    """Used for the skip length in an empty metablock
    >>> skipLengthAlphabet = SkipLengthAlphabet()
    >>> skipLengthAlphabet[0]; str(skipLengthAlphabet[0])
    Symbol(SKIP, 0)
    'empty'
    >>> skipLengthAlphabet[4]
    Traceback (most recent call last):
        ...
    ValueError: index out of range
    >>> print(skipLengthAlphabet[3])
    hhhhhh11
    >>> skipLengthAlphabet[2].value(0x1000)
    4097
    >>> skipLengthAlphabet[3].value(0x1000)
    Traceback (most recent call last):
        ...
    InvalidStream: Zeros in high byte of SKIPBYTES
    >>> skipLengthAlphabet[3].explanation(0x12345)
    'skip length: 12345h+1=74566'
    >>> skipLengthAlphabet.showCode()
    00:empty    01:hh01     10:hhhh10   11:hhhhhh11
    """
    def __init__(self):
        super().__init__('SKIP', bitLength=2)

    def extraBits(self, index):
        return index*8

    def mnemonic(self, index):
        if index==0: return 'empty'
        return 'h'*(self.extraBits(index)//4)+self.bitPattern(index)

    def value(self, index, extra):
        extraBits = self.extraBits(index)
        if not 0<=extra<1<<extraBits:
            raise ValueError('value: extra out of range')
        if index==0: return 0
        if index>1 and extra>>extraBits-8==0:
            raise InvalidStream('Zeros in high byte of SKIPBYTES')
        return extra+1

    def explanation(self, index, extra):
        if index==0: return '00: no skip'
        extraBits = self.extraBits(index)
        return 'skip length: {:{}x}h+1={}'.format(extra, extraBits//8, extra+1)


class TypeCountAlphabet(Enumerator):
    """Used for giving block type counts and tree counts.
    >>> TypeCountAlphabet(description='').showCode()
       0:0            0101:xx,0101      1011:xxxxx,1011
    0001:0001         1101:xxxxxx,1101  0111:xxx,0111
    1001:xxxx,1001    0011:x,0011       1111:xxxxxxx,1111
    """
    decodeTable = {
             0b0: 0,   0b1001: 5,
          0b0001: 1,   0b1011: 6,
          0b0011: 2,   0b1101: 7,
          0b0101: 3,   0b1111: 8,
          0b0111: 4,
          }

    value0 = 1
    extraTable = [0, 0, 1, 2, 3, 4, 5, 6, 7]
    name = 'BT#'

    def __init__(self, name=None, *, description):
        super().__init__(
            name,
            decodeTable=self.decodeTable,
            description=description)

    def mnemonic(self, index):
        if index==0: return '0'
        if index==1: return '0001'
        return 'x'*(self.extraBits(index))+','+self.bitPattern(index)

    def explanation(self, index, extra):
        value = self.value(index, extra)
        description = self.description
        if value==1: description = description[:-1]
        return '{}: {} {}'.format(
            self.mnemonic(index),
            value,
            description)

class BlockTypeAlphabet(Code):
    """The block types; this code works for all three kinds.
    >>> b = BlockTypeAlphabet('T', NBLTYPES=5)
    >>> print(*(x for x in b))
    prev +1 #0 #1 #2 #3 #4
    """
    def __init__(self, name, NBLTYPES, **args):
        super().__init__(name, alphabetSize=NBLTYPES+2, **args)
        self.NBLTYPES = NBLTYPES

    def mnemonic(self, index):
        if index==0: return 'prev'
        elif index==1: return '+1'
        else: return '#'+str(index-2)

    def value(self, index):
        return index-2

    def explanation(self, index):
        if index==0: return '0: previous'
        elif index==1: return '1: increment'
        else: return 'Set block type to: '+str(index-2)

class BlockCountAlphabet(Enumerator):
    """Block counts
    >>> b = BlockCountAlphabet('L')
    >>> print(b[25])
    [24*x]: BC16625-16793840
    """

    value0 = 1
    extraTable = [2,2,2,2,3, 3,3,3,4,4, 4,4,5,5,5, 5,6,6,7,8, 9,10,11,12,13, 24]
    def __init__(self, name, **args):
        super().__init__(name, alphabetSize=26, **args)

    def mnemonic(self, index):
        extraBits = self.extraBits(index)
        return '{}: BC{}-{}'.format(
            'x'*extraBits if index<5 else '[{}*x]'.format(extraBits),
            *self.span(index))

    def explanation(self, index, extra):
        return 'Block count: '+super().explanation(index, extra)

class DistanceParamAlphabet(WithExtra):
    """The distance parameters NPOSTFIX and NDIRECT.
    Although these are treated as two in the description, this is easier.
    """
    def __init__(self):
        super().__init__('DIST', bitLength=2)

    def extraBits(self, index):
        return 4

    def value(self, index, extra):
        """Returns NPOSTFIX and NDIRECT<<NPOSTFIX
        """
        if extra>15:
            raise ValueError('value: extra out of range')
        return index, extra<<index

    def explanation(self, index, extra):
        return '{} postfix bits and {:04b}<<{}={} direct codes'.format(
            index, extra, index, extra<<index)

    def mnemonic(self, index):
        return 'PF'+str(index)

class LiteralContextMode(Code):
    """For the literal context modes.
    >>> LiteralContextMode().showCode()
    00:LSB6   01:MSB6   10:UTF8   11:Signed
    >>> LiteralContextMode().explanation(2)
    'Context mode for type 9: 2(UTF8)'
    """

    def __init__(self, *, number=9):
        super().__init__('LC'+str(number), bitLength=2)
        self.number = number

    def mnemonic(self, index):
        return ['LSB6', 'MSB6', 'UTF8', 'Signed'][index]

    def explanation(self, index):
        return 'Context mode for type {}: {}({})'.format(
            self.number,
            index,
            self.mnemonic(index))

class RLEmaxAlphabet(Enumerator):
    """Used for describing the run length encoding used for describing context maps.
    >>> RLEmaxAlphabet().showCode()
    0:1    1:more
    """
    value0 = 0
    extraTable = [0, 4]
    name = 'RLE#'

    def mnemonic(self, index):
        return ['1', 'more'][index]

    def explanation(self, index, extra):
        description = self.description and self.description+': '
        if index==0: return description+'No RLE coding'
        return '{}xxxx 1: RLEMAX={}'.format(description, extra+1)

class TreeAlphabet(WithExtra):
    """The alphabet to enumerate entries (called trees) in the context map.
    parameters are RLEMAX and NTREES
    >>> t = TreeAlphabet('', RLEMAX=3, NTREES=5)
    >>> len(t)
    8
    >>> print(t[2])
    xx+4 zeroes
    >>> t[3].explanation(2)
    '8+010=10 zeroes'
    >>> t[0].value(0)
    (1, 0)
    """
    name = 'CMI'
    def __init__(self, name=None, *, RLEMAX, NTREES, **args):
        super().__init__(name, alphabetSize=RLEMAX+NTREES, **args)
        self.RLEMAX = RLEMAX
        self.NTREES = NTREES

    def extraBits(self, index):
        if 0<index<=self.RLEMAX: return index
        return 0

    def mnemonic(self, index):
        if index==0: return 'map #0'
        if index<=self.RLEMAX:
            return '{}+{} zeroes'.format('x'*index, 1<<index)
        return 'map #{}'.format(index-self.RLEMAX)

    def value(self, index, extra):
        """Give count and value."""
        index = index
        if index==0: return 1, 0
        if index<=self.RLEMAX: return (1<<index)+extra, 0
        return 1, index-self.RLEMAX

    def explanation(self, index, extra):
        description = self.description and self.description+': '
        if index==0: return description+'map #0'
        if index<=self.RLEMAX:
            return '{}+{:0{}b}={} zeroes'.format(
                (1<<index),
                extra, self.extraBits(index),
                (1<<index)+extra)
        return '{}map #{}-{}={}'.format(
            description,
            index, self.RLEMAX, index-self.RLEMAX)

#Prefix alphabets for the data stream----------------------------------
class LiteralAlphabet(Code):
    """Alphabet of symbols.
    """
    minLength = maxLength = 8
    def __init__(self, number):
        super().__init__('L'+str(number), alphabetSize=1<<8)

    def mnemonic(self, index):
        return outputCharFormatter(index)

    def value(self, index, extra=None):
        return index

    def explanation(self, index, extra=None):
        return self.mnemonic(index)

class InsertLengthAlphabet(Enumerator):
    """Intern code for insert counts
    """
    value0 = 0
    extraTable = [0,0,0,0,0, 0,1,1,2,2, 3,3,4,4,5, 5,6,7,8,9, 10,12,14,24]

class CopyLengthAlphabet(Enumerator):
    value0 = 2
    extraTable = [0,0,0,0,0, 0,0,0,1,1, 2,2,3,3,4, 4,5,5,6,7, 8,9,10,24]

class InsertAndCopyAlphabet(WithExtra):
    """The insert and copy code
    >>> for x in range(0,704,704//13):
    ...    print('{:10b}'.format(x), InsertAndCopyAlphabet()[x])
             0 I0C2&D=0
        110110 I6+xC8&D=0
       1101100 I5C22+xxx&D=0
      10100010 I4C4
      11011000 I3C10+x
     100001110 I14+xxC8
     101000100 I10+xxC22+xxx
     101111010 I98+xxxxxC14+xx
     110110000 I6+xC70+xxxxx
     111100110 I1090+[10*x]C8
    1000011100 I26+xxxC326+[8*x]
    1001010010 I322+[8*x]C14+xx
    1010001000 I194+[7*x]C70+xxxxx
    1010111110 I22594+[24*x]C1094+[10*x]
    """
    insertLengthAlphabet = InsertLengthAlphabet(None)
    copyLengthAlphabet = CopyLengthAlphabet(None)

    def __init__(self, number=''):
        super().__init__('IC'+str(number), bitLength=10)

    def __len__(self):
        return 704

    def extraBits(self, index):
        insertSymbol, copySymbol, dist0 = self.splitSymbol(index)
        return InsertLengthAlphabet.extraTable[insertSymbol.index] + \
            CopyLengthAlphabet.extraTable[copySymbol.index]

    def splitSymbol(self, index):
        """Give relevant values for computations:
        (insertSymbol, copySymbol, dist0flag)
        """
        #determine insert and copy upper bits from table
        row = [0,0,1,1,2,2,1,3,2,3,3][index>>6]
        col = [0,1,0,1,0,1,2,0,2,1,2][index>>6]
        #determine inserts and copy sub codes
        insertLengthCode = row<<3 | index>>3&7
        if row: insertLengthCode -= 8
        copyLengthCode = col<<3 | index&7
        return (
            Symbol(self.insertLengthAlphabet, insertLengthCode),
            Symbol(self.copyLengthAlphabet, copyLengthCode),
            row==0
            )

    def mnemonic(self, index):
        """Make a nice mnemonic
        """
        i,c,d0 = self.splitSymbol(index)
        iLower, _ = i.code.span(i.index)
        iExtra = i.extraBits()
        cLower, _ = c.code.span(c.index)
        cExtra = c.extraBits()
        return 'I{}{}{}C{}{}{}{}'.format(
            iLower,
            '+' if iExtra else '',
            'x'*iExtra if iExtra<6 else '[{}*x]'.format(iExtra),
            cLower,
            '+' if cExtra else '',
            'x'*cExtra if cExtra<6 else '[{}*x]'.format(cExtra),
            '&D=0' if d0 else '')

    def value(self, index, extra):
        i,c,d0 = self.splitSymbol(index)
        iExtra = i.extraBits()
        ce, ie = extra>>iExtra, extra&(1<<iExtra)-1
        insert = i.value(ie)
        copy = c.value(ce)
        return insert, copy, d0

    def explanation(self, index, extra):
        insert, copy, d0 = self.value(index, extra)
        if d0: return 'Literal: {}, copy: {}, same distance'.format(insert, copy)
        else: return 'Literal: {}, copy: {}'.format(insert, copy)

class DistanceAlphabet(WithExtra):
    """Represent the distance encoding.
    Dynamically generated alphabet.
    This is what the documentation should have said:
    Ignoring offsets for the moment, the "long" encoding works as follows:
    Write the distance in binary as follows:
    1xy..yz..z, then the distance symbol consists of n..nxz..z
    Where:
    n is one less than number of bits in y
    x is a single bit
    y..y are n+1 extra bits (encoded in the bit stream)
    z..z is NPOSTFIX bits that are part of the symbol
    The offsets are so as to start at the lowest useable value:
    if 1xyyyyz = distance +(4<<POSTFIX)-NDIRECT-1
    then n..nxz..z is symbol -NDIRECT-16
    >>> d = DistanceAlphabet('D', NPOSTFIX=2, NDIRECT=10)
    >>> print(d[4], d[17], d[34])
    last-1 1 10xx00-5
    >>> [str(d[x]) for x in range(26, 32)]
    ['10x00-5', '10x01-5', '10x10-5', '10x11-5', '11x00-5', '11x01-5']
    """
    def __init__(self, number, *, NPOSTFIX, NDIRECT):
        self.NPOSTFIX = NPOSTFIX
        self.NDIRECT = NDIRECT
        #set length
        #Actually, not all symbols are used,
        #only NDIRECT+16+(44-2*POSTFIX<<NPOSTFIX)
        super().__init__('D'+str(number),
            alphabetSize=self.NDIRECT+16+(48<<self.NPOSTFIX))

    def extraBits(self, index):
        """Indicate how many extra bits are needed to interpret symbol
        >>> d = DistanceAlphabet('D', NPOSTFIX=2, NDIRECT=10)
        >>> [d[i].extraBits() for i in range(26)]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> [d[i].extraBits() for i in range(26,36)]
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
        """
        if index<16+self.NDIRECT: return 0
        return 1 + ((index - self.NDIRECT - 16) >> (self.NPOSTFIX + 1))

    def value(self, dcode, dextra):
        """Decode value of symbol together with the extra bits.
        >>> d = DistanceAlphabet('D', NPOSTFIX=2, NDIRECT=10)
        >>> d[34].value(2)
        (0, 35)
        """
        if dcode<16:
            return [(1,0),(2,0),(3,0),(4,0),
                    (1,-1),(1,+1),(1,-2),(1,+2),(1,-3),(1,+3),
                    (2,-1),(2,+1),(2,-2),(2,+2),(2,-3),(2,+3)
                ][dcode]
        if dcode<16+self.NDIRECT:
            return (0,dcode-16)
        #we use the original formulas, instead of my clear explanation
        POSTFIX_MASK = (1 << self.NPOSTFIX) - 1
        ndistbits = 1 + ((dcode - self.NDIRECT - 16) >> (self.NPOSTFIX + 1))
        hcode = (dcode - self.NDIRECT - 16) >> self.NPOSTFIX
        lcode = (dcode - self.NDIRECT - 16) & POSTFIX_MASK
        offset = ((2 + (hcode & 1)) << ndistbits) - 4
        distance = ((offset + dextra) << self.NPOSTFIX) + lcode + self.NDIRECT + 1
        return (0,distance)

    def mnemonic(self, index, verbose=False):
        """Give mnemonic representation of meaning.
        verbose compresses strings of x's
        """
        if index<16:
            return ['last', '2last', '3last', '4last',
                'last-1', 'last+1', 'last-2', 'last+2', 'last-3', 'last+3',
                '2last-1', '2last+1', '2last-2', '2last+2', '2last-3', '2last+3'
                ][index]
        if index<16+self.NDIRECT:
            return str(index-16)
        #construct strings like "1xx01-15"
        index -= self.NDIRECT+16
        hcode = index >> self.NPOSTFIX
        lcode = index & (1<<self.NPOSTFIX)-1
        if self.NPOSTFIX: formatString = '1{0}{1}{2:0{3}b}{4:+d}'
        else: formatString = '1{0}{1}{4:+d}'
        return formatString.format(
            hcode&1,
            'x'*(2+hcode>>1) if hcode<13 or verbose else '[{}*x]'.format(2+hcode>>1),
            lcode, self.NPOSTFIX,
            self.NDIRECT+1-(4<<self.NPOSTFIX))

    def explanation(self, index, extra):
        """
        >>> d = DistanceAlphabet('D', NPOSTFIX=2, NDIRECT=10)
        >>> d[55].explanation(13)
        '11[1101]01-5: [0]+240'
        """
        extraBits = self.extraBits(index)
        extraString = '[{:0{}b}]'.format(extra, extraBits)
        return '{0}: [{1[0]}]{1[1]:+d}'.format(
            self.mnemonic(index, True).replace('x'*(extraBits or 1), extraString),
            self.value(index, extra))

#Classes for doing actual work------------------------------------------
class ContextModeKeeper:
    """For computing the literal context mode.
    You feed it characters, and it computes indices in the context map.
    """
    def __init__(self, mode):
        self.chars = deque([0,0], maxlen=2)
        self.mode = mode

    def setContextMode(self, mode):
        """Switch to given context mode (0..3)"""
        self.mode = mode
    def getIndex(self):
        if self.mode==0:  #LSB6
            return self.chars[1]&0x3f
        elif self.mode==1: #MSB6
            return self.chars[1]>>2
        elif self.mode==2: #UTF8: character class of previous and a bit of the second
            p2,p1 = self.chars
            return self.lut0[p1]|self.lut1[p2]
        elif self.mode==3: #Signed: initial bits of last two bytes
            p2,p1 = self.chars
            return self.lut2[p1]<<3|self.lut2[p2]

    def add(self, index):
        """Adjust the context for output char (as int)."""
        self.chars.append(index)

    #0: control     #16: quote  #32: ,:;  #48: AEIOU
    #4: tab/lf/cr   #20: %      #36: .    #52: BC..Z
    #8: space       #24: (<[{   #40: =    #56: aeiou
    #12:!#$&*+-/?@| #28: )>]}   #44: 0-9  #60: bc..z
    lut0 = [0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  4,  0,  0,  4,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            8, 12, 16, 12, 12, 20, 12, 16, 24, 28, 12, 12, 32, 12, 36, 12,
           44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 32, 32, 24, 40, 28, 12,
           12, 48, 52, 52, 52, 48, 52, 52, 52, 48, 52, 52, 52, 52, 52, 48,
           52, 52, 52, 52, 52, 48, 52, 52, 52, 52, 52, 24, 12, 28, 12, 12,
           12, 56, 60, 60, 60, 56, 60, 60, 60, 56, 60, 60, 60, 60, 60, 56,
           60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 24, 12, 28, 12,  0
           ]+[0,1]*32+[2,3]*32
    #0: space  1:punctuation  2:digit/upper 3:lower
    lut1 = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
             1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
             1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 0
           ]+[0]*96+[2]*32
    #initial bits: 8*0, 4*0, 2*0, 1*0, 1*1, 2*1, 4*1, 8*1
    lut2 = [0]+[1]*15+[2]*48+[3]*64+[4]*64+[5]*48+[6]*15+[7]
    assert len(lut0)==len(lut1)==len(lut2)==256

class WordList:
    """Word list.
    >>> WordList().word(7, 35555)
    b'Program to '
    """
    NDBITS = [0,  0,  0,  0, 10, 10, 11, 11, 10, 10,
             10, 10, 10,  9,  9,  8,  7,  7,  8,  7,
              7,  6,  6,  5,  5]
    def __init__(self):
        self.file = open('dict', 'rb')
        self.compileActions()

    def word(self, size, dist):
        """Get word
        """
        #split dist in index and action
        ndbits = self.NDBITS[size]
        index = dist&(1<<ndbits)-1
        action = dist>>ndbits
        #compute position in file
        position = sum(n<<self.NDBITS[n] for n in range(4,size))+size*index
        self.file.seek(position)
        return self.doAction(self.file.read(size), action)

    def upperCase1(self, word):
        word = word.decode('utf8')
        word = word[0].upper()+word[1:]
        return word.encode('utf8')


    #Super compact form of action table.
    #_ means space, .U means UpperCaseAll, U(w) means UpperCaseFirst
    actionTable = r"""
        0:w        25:w+_for_     50:w+\n\t       75:w+. This_100:w+ize_
        1:w+_      26:w[3:]       51:w+:          76:w+,      101:w.U+.
        2:_+w+_    27:w[:-2]      52:_+w+._       77:.+w+_    102:\xc2\xa0+w
        3:w[1:]    28:w+_a_       53:w+ed_        78:U(w)+(   103:_+w+,
        4:U(w)+_   29:w+_that_    54:w[9:]        79:U(w)+.   104:U(w)+="
        5:w+_the_  30:_+U(w)      55:w[7:]        80:w+_not_  105:w.U+="
        6:_+w      31:w+._        56:w[:-6]       81:_+w+="   106:w+ous_
        7:s_+w+_   32:.+w         57:w+(          82:w+er_    107:w.U+,_
        8:w+_of_   33:_+w+,_      58:U(w)+,_      83:_+w.U+_  108:U(w)+=\'
        9:U(w)     34:w[4:]       59:w[:-8]       84:w+al_    109:_+U(w)+,
       10:w+_and_  35:w+_with_    60:w+_at_       85:_+w.U    110:_+w.U+="
       11:w[2:]    36:w+\'        61:w+ly_        86:w+=\'    111:_+w.U+,_
       12:w[:-1]   37:w+_from_    62:_the_+w+_of_ 87:w.U+"    112:_+w.U+,
       13:,_+w+_   38:w+_by_      63:w[:-5]       88:U(w)+._  113:w.U+(
       14:w+,_     39:w[5:]       64:w[:-9]       89:_+w+(    114:w.U+._
       15:_+U(w)+_ 40:w[6:]       65:_+U(w)+,_    90:w+ful_   115:_+w.U+.
       16:w+_in_   41:_the_+w     66:U(w)+"       91:_+U(w)+._116:w.U+=\'
       17:w+_to_   42:w[:-4]      67:.+w+(        92:w+ive_   117:_+w.U+._
       18:e_+w+_   43:w+. The_    68:w.U+_        93:w+less_  118:_+U(w)+="
       19:w+"      44:w.U         69:U(w)+">      94:w.U+\'   119:_+w.U+=\'
       20:w+.      45:w+_on_      70:w+="         95:w+est_   120:_+U(w)+=\'
       21:w+">     46:w+_as_      71:_+w+.        96:_+U(w)+.
       22:w+\n     47:w+_is_      72:.com/+w      97:w.U+">
       23:w[:-3]   48:w[:-7]                      98:_+w+=\'
       24:w+]      49:w[:-1]+ing_ 74:U(w)+\'      99:U(w)+,
        """

    def compileActions(self):
        """Build the action table from the text above
        """
        import re
        self.actionList = actions = [None]*121
        #Action 73, which is too long, looks like this when expanded:
        actions[73] = "b' the '+w+b' of the '"
        #find out what the columns are
        actionLines = self.actionTable.splitlines()
        colonPositions = [m.start()
            for m in re.finditer(':',actionLines[1])
            ]+[100]
        columns = [(colonPositions[i]-3,colonPositions[i+1]-3)
            for i in range(len(colonPositions)-1)]
        for line in self.actionTable.splitlines(keepends=False):
            for start,end in columns:
                action = line[start:end]
                #skip empty actions
                if not action or action.isspace(): continue
                #chop it up, and check if the colon is properly placed
                index, colon, action = action[:3], action[3], action[4:]
                assert colon==':'
                #remove filler spaces at right
                action = action.rstrip()
                #replace space symbols
                action = action.replace('_', ' ')
                wPos = action.index('w')
                #add quotes around left string when present
                #translation: any pattern from beginning, up to
                #(but not including) a + following by a w later on
                action = re.sub(r"^(.*)(?=\+[U(]*w)", r"b'\1'", action)
                #add quotes around right string when present
                #translation: anything with a w in it, followed by a +
                #and a pattern up to the end
                #(there is no variable lookbehind assertion,
                #so we have to copy the pattern)
                action = re.sub(r"(w[[:\-1\]).U]*)\+(.*)$", r"\1+b'\2'", action)
                #expand shortcut for uppercaseAll
                action = action.replace(".U", ".upper()")
                #store action
                actions[int(index)] = action

    def doAction(self, w, action):
        """Perform the proper action
        """
        #set environment for the UpperCaseFirst
        U = self.upperCase1
        return eval(self.actionList[action], locals())

class Layout:
    """Class to layout the output.
    """
    #display width of hexdata+bitdata
    width = 25
    #general
    def __init__(self, stream):
        self.stream = stream
        self.bitPtr = self.width

    def makeHexData(self, pos):
        """Produce hex dump of all data containing the bits
        from pos to stream.pos
        """
        firstAddress = pos+7>>3
        lastAddress = self.stream.pos+7>>3
        return ''.join(map('{:02x} '.format,
            self.stream.data[firstAddress:lastAddress]))

    def formatBitData(self, pos, width1, width2=0):
        """Show formatted bit data:
        Bytes are separated by commas
        whole bytes are displayed in hex
        >>> Layout(olleke).formatBitData(6, 2, 16)
        '|00h|2Eh,|00'
        >>> Layout(olleke).formatBitData(4, 1, 0)
        '1'
        """
        result = []
        #make empty prefix code explicit
        if width1==0: result = ['()', ',']
        for width in width1, width2:
            #skip empty width2
            if width==0: continue
            #build result backwards in a list
            while width>0:
                availableBits = 8-(pos&7)
                if width<availableBits:
                    #read partial byte, beginning nor ending at boundary
                    data = self.stream.data[pos>>3] >> (pos&7) & (1<<width)-1
                    result.append('{:0{}b}'.format(data, width))
                elif availableBits<8:
                    #read rest of byte, ending at boundary
                    data = self.stream.data[pos>>3] >> (pos&7)
                    result.append('|{:0{}b}'.format(data, availableBits))
                else:
                    #read whole byte (in hex), beginning and ending at boundary
                    data = self.stream.data[pos>>3]
                    result.append('|{:02X}h'.format(data))
                width -= availableBits
                pos += availableBits
            #if width overshot from the availableBits subtraction, fix it
            pos += width
            #add comma to separate fields
            result.append(',')
        #concatenate pieces, reversed, skipping the last space
        return ''.join(result[-2::-1])

    def readPrefixCode(self, alphabet):
        """give alphabet the prefix code that is read from the stream
        Called for the following alphabets, in this order:
        The alphabet in question must have a "logical" order,
        otherwise the assignment of symbols doesn't work.
        """
        mode, numberOfSymbols = self.verboseRead(PrefixCodeHeader(alphabet.name))
        if mode=='Complex':
            #for a complex code, numberOfSymbols means hskip
            self.readComplexCode(numberOfSymbols, alphabet)
            return alphabet
        else:
            table = []
            #Set table of lengths for mnemonic function
            lengths = [[0], [1,1], [1,2,2], '????'][numberOfSymbols-1]
            #adjust mnemonic function of alphabet class
            def myMnemonic(index):
                return '{} bit{}: {}'.format(
                    lengths[i],
                    '' if lengths[i]==1 else 's',
                    alphabet.__class__.mnemonic(alphabet, index)
                    )
            alphabet.mnemonic = myMnemonic
            for i in range(numberOfSymbols):
                table.append(self.verboseRead(alphabet, skipExtra=True).index)
            #restore mnemonic
            del alphabet.mnemonic
            if numberOfSymbols==4:
                #read tree shape to redefine lengths
                lengths = self.verboseRead(TreeShapeAlhabet())
            #construct the alphabet prefix code
            alphabet.setLength(dict(zip(table, lengths)))
        return alphabet

    def readComplexCode(self, hskip, alphabet):
        """Read complex code"""
        stream = self.stream
        #read the lengths for the length code
        lengths = [1,2,3,4,0,5,17,6,16,7,8,9,10,11,12,13,14,15][hskip:]
        codeLengths = {}
        total = 0
        lol = LengthOfLengthAlphabet('##'+alphabet.name)
        #lengthCode will be used for coding the lengths of the new code
        #we use it for display until now; definition comes below
        lengthCode = LengthAlphabet('#'+alphabet.name)
        lengthIter = iter(lengths)
        lengthsLeft = len(lengths)
        while total<32 and lengthsLeft>0:
            lengthsLeft -= 1
            newSymbol = next(lengthIter)
            lol.description = str(lengthCode[newSymbol])
            length = self.verboseRead(lol)
            if length:
                codeLengths[newSymbol] = length
                total += 32>>length
        if total>32: raise ValueError("Stream format")
        if len(codeLengths)==1: codeLengths[list(codeLengths.keys())[0]] = 0
        #Now set the encoding of the lengthCode
        lengthCode.setLength(codeLengths)
        print("***** Lengths for {} will be coded as:".format(alphabet.name))
        lengthCode.showCode()
        #Now determine the symbol lengths with the lengthCode
        symbolLengths = {}
        total = 0
        lastLength = 8
        alphabetIter = iter(alphabet)
        while total<32768:
            #look ahead to see what is going to happen
            length = lengthCode.decodePeek(
                self.stream.peek(lengthCode.maxLength))[1].index
            #in every branch, set lengthCode.description to explanatory text
            #lengthCode calls format(symbol, extra) with this string
            if length==0:
                symbol = next(alphabetIter)
                lengthCode.description = 'symbol {} unused'.format(symbol)
                self.verboseRead(lengthCode)
                #unused symbol
                continue
            if length==16:
                lengthCode.description = \
                    '{1}+3 symbols of length '+str(lastLength)
                extra = self.verboseRead(lengthCode)
                #scan series of 16s (repeat counts)
                #start with repeat count 2
                repeat = 2
                startSymbol = next(alphabetIter)
                endSymbol = next(alphabetIter)
                symbolLengths[startSymbol.index] = \
                    symbolLengths[endSymbol.index] = lastLength
                #count the two just defined symbols
                total += 2*32768>>lastLength
                #note: loop may end because we're there
                #even if a 16 _appears_ to follow
                while True:
                    #determine last symbol
                    oldRepeat = repeat
                    repeat = (repeat-2<<2)+extra+3
                    #read as many symbols as repeat increased
                    for i in range(oldRepeat, repeat):
                        endSymbol = next(alphabetIter)
                        symbolLengths[endSymbol.index] = lastLength
                    #compute new total; it may be end of loop
                    total += (repeat-oldRepeat)*32768>>lastLength
                    if total>=32768: break
                    #see if there is more to do
                    length = lengthCode.decodePeek(
                        self.stream.peek(lengthCode.maxLength))[1].index
                    if length!=16: break
                    lengthCode.description = 'total {}+{{1}} symbols'.format(
                        (repeat-2<<2)+3)
                    extra = self.verboseRead(lengthCode)
            elif length==17:
                #read, and show explanation
                lengthCode.description = '{1}+3 unused'
                extra = self.verboseRead(lengthCode)
                #scan series of 17s (groups of zero counts)
                #start with repeat count 2
                repeat = 2
                startSymbol = next(alphabetIter)
                endSymbol = next(alphabetIter)
                #note: loop will not end with total==32768,
                #since total doesn't change here
                while True:
                    #determine last symbol
                    oldRepeat = repeat
                    repeat = (repeat-2<<3)+extra+3
                    #read as many symbols as repeat increases
                    for i in range(repeat-oldRepeat):
                        endSymbol = next(alphabetIter)
                    #see if there is more to do
                    length = lengthCode.decodePeek(
                        self.stream.peek(lengthCode.maxLength))[1].index
                    if length!=17: break
                    lengthCode.description = 'total {}+{{1}} unused'.format(
                        (repeat-2<<3)+3)
                    extra = self.verboseRead(lengthCode)
            else:
                symbol = next(alphabetIter)
                #double braces for format
                char = str(symbol)
                if char in '{}': char *= 2
                lengthCode.description = \
                    'Length for {} is {{0.index}} bits'.format(char)
                #output is not needed (will be 0)
                self.verboseRead(lengthCode)
                symbolLengths[symbol.index] = length
                total += 32768>>length
                lastLength = length
        assert total==32768
        alphabet.setLength(symbolLengths)
        print('End of table. Prefix code '+alphabet.name+':')
        alphabet.showCode()

    #stream
    def processStream(self):
        """Process a brotli stream.
        """
        print('addr  hex{:{}s}binary context explanation'.format(
            '', self.width-10))
        print('Stream header'.center(60, '-'))
        self.windowSize = self.verboseRead(WindowSizeAlphabet())
        print('Metablock header'.center(60, '='))
        self.ISLAST = False
        self.output = bytearray()
        while not self.ISLAST:
            self.ISLAST = self.verboseRead(
                BoolCode('LAST', description="Last block"))
            if self.ISLAST:
                if self.verboseRead(
                    BoolCode('EMPTY', description="Empty block")): break
            if self.metablockLength(): continue
            if not self.ISLAST and self.uncompressed(): continue
            print('Block type descriptors'.center(60, '-'))
            self.numberOfBlockTypes = {}
            self.currentBlockCounts = {}
            self.blockTypeCodes = {}
            self.blockCountCodes = {}
            for blockType in (L,I,D): self.blockType(blockType)
            print('Distance code parameters'.center(60, '-'))
            self.NPOSTFIX, self.NDIRECT = self.verboseRead(DistanceParamAlphabet())
            self.readLiteralContextModes()
            print('Context maps'.center(60, '-'))
            self.cmaps = {}
            #keep the number of each kind of prefix tree for the last loop
            numberOfTrees = {I: self.numberOfBlockTypes[I]}
            for blockType in (L,D):
                numberOfTrees[blockType] = self.contextMap(blockType)
            print('Prefix code lists'.center(60, '-'))
            self.prefixCodes = {}
            for blockType in (L,I,D):
                self.readPrefixArray(blockType, numberOfTrees[blockType])
            self.metablock()

    #metablock header
    def verboseRead(self, alphabet, context='', skipExtra=False):
        """Read symbol and extra from stream and explain what happens.
        Returns the value of the symbol
        >>> olleke.pos = 0
        >>> l = Layout(olleke)
        >>> l.verboseRead(WindowSizeAlphabet())
        0000  1b                   1011 WSIZE   windowsize=(1<<22)-16=4194288
        4194288
        """
        #TODO 2: verbosity level, e.g. show only codes and maps in header
        stream = self.stream
        pos = stream.pos
        if skipExtra:
            length, symbol = alphabet.readTuple(stream)
            extraBits, extra = 0, None
        else:
            length, symbol, extraBits, extra = alphabet.readTupleAndExtra(
                stream)
        #fields: address, hex data, binary data, name of alphabet, explanation
        hexdata = self.makeHexData(pos)
        addressField = '{:04x}'.format(pos+7>>3) if hexdata else ''
        bitdata = self.formatBitData(pos, length, extraBits)
        #bitPtr moves bitdata so that the bytes are easier to read
        #jump back to right if a new byte starts
        if '|' in bitdata[1:]:
            #start over on the right side
            self.bitPtr = self.width
        fillWidth = self.bitPtr-(len(hexdata)+len(bitdata))
        if fillWidth<0: fillWidth = 0
        print('{:<5s} {:<{}s} {:7s} {}'.format(
            addressField,
            hexdata+' '*fillWidth+bitdata, self.width,
            context+alphabet.name,
            symbol if skipExtra else symbol.explanation(extra),
            ))
        #jump to the right if we started with a '|'
        #because we didn't jump before printing
        if bitdata.startswith('|'): self.bitPtr = self.width
        else: self.bitPtr -= len(bitdata)
        return symbol if skipExtra else symbol.value(extra)

    def metablockLength(self):
        """Read MNIBBLES and meta block length;
        if empty block, skip block and return true.
        """
        self.MLEN = self.verboseRead(MetablockLengthAlphabet())
        if self.MLEN:
            return False
        #empty block; skip and return False
        self.verboseRead(ReservedAlphabet())
        MSKIP = self.verboseRead(SkipLengthAlphabet())
        self.verboseRead(FillerAlphabet(streamPos=self.stream.pos))
        self.stream.pos += 8*MSKIP
        print("Skipping to {:x}".format(self.stream.pos>>3))
        return True

    def uncompressed(self):
        """If true, handle uncompressed data
        """
        ISUNCOMPRESSED = self.verboseRead(
            BoolCode('UNCMPR', description='Is uncompressed?'))
        if ISUNCOMPRESSED:
            self.verboseRead(FillerAlphabet(streamPos=self.stream.pos))
            print('Uncompressed data:')
            self.output += self.stream.readBytes(self.MLEN)
            print(outputFormatter(self.output[-self.MLEN:]))
        return ISUNCOMPRESSED

    def blockType(self, kind):
        """Read block type switch descriptor for given kind of blockType."""
        NBLTYPES = self.verboseRead(TypeCountAlphabet(
            'BT#'+kind[0].upper(),
            description='{} block types'.format(kind),
            ))
        self.numberOfBlockTypes[kind] = NBLTYPES
        if NBLTYPES>=2:
            self.blockTypeCodes[kind] = self.readPrefixCode(
                BlockTypeAlphabet('BT'+kind[0].upper(), NBLTYPES))
            self.blockCountCodes[kind] = self.readPrefixCode(
                BlockCountAlphabet('BC'+kind[0].upper()))
            blockCount = self.verboseRead(self.blockCountCodes[kind])
        else:
            blockCount = 1<<24
        self.currentBlockCounts[kind] = blockCount

    def readLiteralContextModes(self):
        """Read literal context modes.
        LSB6: lower 6 bits of last char
        MSB6: upper 6 bits of last char
        UTF8: rougly dependent on categories:
            upper 4 bits depend on category of last char:
                control/whitespace/space/ punctuation/quote/%/open/close/
                comma/period/=/digits/ VOWEL/CONSONANT/vowel/consonant
            lower 2 bits depend on category of 2nd last char:
                space/punctuation/digit or upper/lowercase
        signed: hamming weight of last 2 chars
        """
        print('Context modes'.center(60, '-'))
        self.literalContextModes = []
        for i in range(self.numberOfBlockTypes[L]):
            self.literalContextModes.append(
                self.verboseRead(LiteralContextMode(number=i)))

    def contextMap(self, kind):
        """Read context maps
        Returns the number of differnt values on the context map
        (In other words, the number of prefix trees)
        """
        NTREES = self.verboseRead(TypeCountAlphabet(
            kind[0].upper()+'T#',
            description='{} prefix trees'.format(kind)))
        mapSize = {L:64, D:4}[kind]
        if NTREES<2:
            self.cmaps[kind] = [0]*mapSize
        else:
            #read CMAPkind
            RLEMAX = self.verboseRead(RLEmaxAlphabet(
                'RLE#'+kind[0].upper(),
                description=kind+' context map'))
            alphabet = TreeAlphabet('CM'+kind[0].upper(), NTREES=NTREES, RLEMAX=RLEMAX)
            cmapCode = self.readPrefixCode(alphabet)
            tableSize = mapSize*self.numberOfBlockTypes[kind]
            cmap = []
            while len(cmap)<tableSize:
                cmapCode.description = 'map {}, entry {}'.format(
                    *divmod(len(cmap), mapSize))
                count, value = self.verboseRead(cmapCode)
                cmap.extend([value]*count)
            assert len(cmap)==tableSize
            IMTF = self.verboseRead(BoolCode('IMTF', description='Apply inverse MTF'))
            if IMTF:
                self.IMTF(cmap)
            if kind==L:
                print('Context maps for literal data:')
                for i in range(0, len(cmap), 64):
                    print(*(
                        ''.join(map(str, cmap[j:j+8]))
                        for j in range(i, i+64, 8)
                        ))
            else:
                print('Context map for distances:')
                print(*(
                    ''.join(map('{:x}'.format, cmap[i:i+4]))
                    for i in range(0, len(cmap), 4)
                    ))
            self.cmaps[kind] = cmap
        return NTREES

    @staticmethod
    def IMTF(v):
        """In place inverse move to front transform.
        """
        #mtf is initialized virtually with range(infinity)
        mtf = []
        for i, vi in enumerate(v):
            #get old value from mtf. If never seen, take virtual value
            try: value = mtf.pop(vi)
            except IndexError: value = vi
            #put value at front
            mtf.insert(0, value)
            #replace transformed value
            v[i] = value

    def readPrefixArray(self, kind, numberOfTrees):
        """Read prefix code array"""
        prefixes = []
        for i in range(numberOfTrees):
            if kind==L: alphabet = LiteralAlphabet(i)
            elif kind==I: alphabet = InsertAndCopyAlphabet(i)
            elif kind==D: alphabet = DistanceAlphabet(
                i, NPOSTFIX=self.NPOSTFIX, NDIRECT=self.NDIRECT)
            self.readPrefixCode(alphabet)
            prefixes.append(alphabet)
        self.prefixCodes[kind] = prefixes

    #metablock data
    def metablock(self):
        """Process the data.
        Relevant variables of self:
        numberOfBlockTypes[kind]: number of block types
        currentBlockTypes[kind]: current block types (=0)
        literalContextModes: the context modes for the literal block types
        currentBlockCounts[kind]: counters for block types
        blockTypeCodes[kind]: code for block type
        blockCountCodes[kind]: code for block count
        cmaps[kind]: the context maps (not for I)
        prefixCodes[kind][#]: the prefix codes
        lastDistances: the last four distances
        lastChars: the last two chars
        output: the result
        """
        print('Meta block contents'.center(60, '='))
        self.currentBlockTypes = {L:0, I:0, D:0, pL:1, pI:1, pD:1}
        self.lastDistances = deque([17,16,11,4], maxlen=4)
        #the current context mode is for block type 0
        self.contextMode = ContextModeKeeper(self.literalContextModes[0])
        wordList = WordList()

        #setup distance callback function
        def distanceCallback(symbol, extra):
            "callback function for displaying decoded distance"
            index, offset = symbol.value(extra)
            if index:
                #recent distance
                distance = self.lastDistances[-index]+offset
                return 'Distance: {}last{:+d}={}'.format(index, offset, distance)
            #absolute value
            if offset<=maxDistance:
                return 'Absolute value: {} (pos {})'.format(offset, maxDistance-offset)
            #word list value
            action, word = divmod(offset-maxDistance, 1<<wordList.NDBITS[copyLen])
            return '{}-{} gives word {},{} action {}'.format(
                offset, maxDistance, copyLen, word, action)
        for dpc in self.prefixCodes[D]: dpc.callback = distanceCallback

        blockLen = 0
        #there we go
        while blockLen<self.MLEN:
            #get insert&copy command
            litLen, copyLen, dist0Flag = self.verboseRead(
                self.prefixCodes[I][
                    self.figureBlockType(I)])
            #literal data
            for i in range(litLen):
                bt = self.figureBlockType(L)
                cm = self.contextMode.getIndex()
                ct = self.cmaps[L][bt<<6|cm]
                char = self.verboseRead(
                    self.prefixCodes[L][ct],
                    context='{},{}='.format(bt,cm))
                self.contextMode.add(char)
                self.output.append(char)
            blockLen += litLen
            #check if we're done
            if blockLen>=self.MLEN: return
            #distance
            #distances are computed relative to output length, at most window size
            maxDistance = min(len(self.output), self.windowSize)
            if dist0Flag:
                distance = self.lastDistances[-1]
            else:
                bt = self.figureBlockType(D)
                cm = {2:0, 3:1, 4:2}.get(copyLen, 3)
                ct = self.cmaps[D][bt<<2|cm]
                index, offset = self.verboseRead(
                    self.prefixCodes[D][ct],
                    context='{},{}='.format(bt,cm))
                distance = self.lastDistances[-index]+offset if index else offset
                if index==1 and offset==0:
                    #to make sure distance is not put in last distance list
                    dist0Flag = True
            if distance<=maxDistance:
                #copy from output
                for i in range(
                        maxDistance-distance,
                        maxDistance-distance+copyLen):
                    self.output.append(self.output[i])
                if not dist0Flag: self.lastDistances.append(distance)
                comment = 'Seen before'
            else:
                #fetch from wordlist
                newWord = wordList.word(copyLen, distance-maxDistance-1)
                self.output.extend(newWord)
                #adjust copyLen to reflect actual new data
                copyLen = len(newWord)
                comment = 'From wordlist'
            blockLen += copyLen
            print(' '*40,
                comment,
                ': "',
                outputFormatter(self.output[-copyLen:]),
                '"',
                sep='')
            self.contextMode.add(self.output[-2])
            self.contextMode.add(self.output[-1])

    def figureBlockType(self, kind):
        counts, types = self.currentBlockCounts, self.currentBlockTypes
        if counts[kind]==0:
            newType = self.verboseRead(self.blockTypeCodes[kind])
            if newType==-2: newType = types['P'+kind]
            elif newType==-1:
                newType = (types[kind]+1)%self.numberOfBlockTypes[kind]
            types['P'+kind] = types[kind]
            types[kind] = newType
            counts[kind] = self.verboseRead(self.blockCountCodes[kind])
        counts[kind] -=1
        return types[kind]

__test__ = {
'BitStream': """
    >>> bs = BitStream(b'Jurjen')
    >>> bs.readBytes(2)
    b'Ju'
    >>> bs.read(6) #r=01110010
    50
    >>> bs
    BitStream(pos=2:6)
    >>> bs.peek(5)  #j=01101010
    9
    >>> bs.readBytes(2)
    Traceback (most recent call last):
        ...
    ValueError: readBytes: need byte boundary
    """,

'Symbol': """
    >>> a=Symbol(MetablockLengthAlphabet(),5)
    >>> len(a)
    2
    >>> int(a)
    5
    >>> a.bitPattern()
    '01'
    >>> a.value(200000)
    200001
    >>> a.explanation(300000)
    'data length: 493e0h+1=300001'
    """,

'RangeDecoder': """
    >>> a=RangeDecoder(bitLength=3)
    >>> len(a)
    8
    >>> a.name='t'
    >>> list(a)
    [Symbol(t, 0), Symbol(t, 1), Symbol(t, 2), Symbol(t, 3), Symbol(t, 4), Symbol(t, 5), Symbol(t, 6), Symbol(t, 7)]
    >>> a[2]
    Symbol(t, 2)
    >>> a.bitPattern(4)
    '100'
    >>> a.length(2)
    3
    >>> a.decodePeek(15)
    (3, Symbol(t, 7))
    >>>

    """,

'PrefixDecoder': """
    >>> a=PrefixDecoder(decodeTable={0:1,1:2,3:3,7:4})
    >>> len(a)
    4
    >>> a.name='t'
    >>> list(a)
    [Symbol(t, 1), Symbol(t, 2), Symbol(t, 3), Symbol(t, 4)]
    >>> a.decodePeek(22)
    (1, Symbol(t, 1))
    >>> a.decodePeek(27)
    (3, Symbol(t, 3))
    >>> a.length(1)
    1
    >>> a.length(4)
    3
    """,

'Code': """
    >>> a=Code('t',alphabetSize=10)
    >>> len(a)
    10
    >>> a.showCode()
    0000:0 0001:1 0010:2 0011:3 0100:4 0101:5 0110:6 0111:7 1000:8 1001:9
    >>> a.setLength({2:1,3:2,5:3,6:3})
    >>> a.showCode()
      0:2  01:3 011:5 111:6
    >>> len(a)
    4
    >>> def callback(i): return 'call{}back'.format(i)
    >>> a=Code('t',callback=callback,bitLength=3)
    >>> a[6].explanation()
    'call6back'
    """,

'WithExtra': """
    >>> class A(WithExtra):
    ...    extraTable = [0,1,1,2,2]
    >>> a=A('t',alphabetSize=5)
    >>> a[1]
    Symbol(t, 1)
    >>> a.extraBits(2)
    1
    >>> a.mnemonic(4)
    '4'
    >>> a.readTupleAndExtra(BitStream(b'\x5b'))
    (3, Symbol(t, 3), 2, 3)
    """,

'BoolCode': """
    >>> BoolCode('test')[0].explanation()
    '0: False'
    """,

'Enumerator': """
    >>> class A(Enumerator):
    ...    extraTable = [0,1,1,2,2]
    ...    value0=3
    >>> a=A(alphabetLength=5)
    >>> a.value(3)
    Traceback (most recent call last):
        ...
    TypeError: value() missing 1 required positional argument: 'extra'
    >>> a.explanation(3,4)
    'xx 011: 8-11; 8+4=12'
    """,

'WindowSizeAlphabet': """
    >>> windowSizeAlphabet = WindowSizeAlphabet()
    >>> windowSizeAlphabet[0]
    Traceback (most recent call last):
        ...
    ValueError: No symbol WindowSizeAlphabet[0]
    >>> len(windowSizeAlphabet)
    16
    >>> windowSizeAlphabet[21]
    Symbol(WSIZE, 21)
    >>> windowSizeAlphabet[21].bitPattern()
    '1001'
    >>> windowSizeAlphabet[21].extraBits()
    0
    >>> windowSizeAlphabet[21].index
    21
    >>> windowSizeAlphabet[10].value()
    1008
    >>> windowSizeAlphabet[10].explanation()
    'windowsize=(1<<10)-16=1008'
    >>> windowSizeAlphabet.showCode()
          0:65520    1100001:16368    1110001:32752       0011:262128
    0000001:131056   0010001:None        1001:2097136     1011:4194288
    1000001:4080     1010001:8176        0101:524272      0111:1048560
    0100001:1008     0110001:2032        1101:8388592     1111:16777200
    """,

'TypeCountAlphabet': """
    >>> typeCountAlphabet = TypeCountAlphabet(description='bananas')
    >>> len(typeCountAlphabet)
    9
    >>> typeCountAlphabet[3]
    Symbol(BT#, 3)
    >>> typeCountAlphabet[9]
    Traceback (most recent call last):
        ...
    ValueError: No symbol TypeCountAlphabet[9]
    >>> print(typeCountAlphabet[3])
    xx,0101
    >>> typeCountAlphabet[8].value(127)
    256
    >>> typeCountAlphabet[4].explanation(2)
    'xxx,0111: 11 bananas'
    >>> typeCountAlphabet[0].explanation()
    '0: 1 banana'
    """,

'DistanceParamAlphabet': """
    >>> dpa = DistanceParamAlphabet()
    >>> dpa.showCode()
    00:PF0 01:PF1 10:PF2 11:PF3
    >>> dpa.readTupleAndExtra(BitStream(b'\\x29'))
    (2, Symbol(DIST, 1), 4, 10)
    >>> dpa.explanation(2, 5)
    '2 postfix bits and 0101<<2=20 direct codes'
    """,

'LiteralAlphabet': """
    >>> LiteralAlphabet(-1).showCode()   #doctest: +ELLIPSIS
    00000000:\\x00 00110100:4    01101000:h    10011100:\\x9c 11010000:\\xd0
    00000001:\\x01 00110101:5    01101001:i    10011101:\\x9d 11010001:\\xd1
    00000010:\\x02 00110110:6    01101010:j    10011110:\\x9e 11010010:\\xd2
    ...
    00101111:/    01100011:c    10010111:\\x97 11001011:\\xcb 11111111:\\xff
    00110000:0    01100100:d    10011000:\\x98 11001100:\\xcc
    00110001:1    01100101:e    10011001:\\x99 11001101:\\xcd
    00110010:2    01100110:f    10011010:\\x9a 11001110:\\xce
    00110011:3    01100111:g    10011011:\\x9b 11001111:\\xcf
    """,

'BlockCountAlphabet': """
    >>> bc=BlockCountAlphabet('BCL')
    >>> len(bc)
    26
    >>> bs=BitStream(b'\\x40\\x83\\xc8\\x59\\12\\x02')
    >>> x = bc.readTupleAndExtra(bs); x[1].explanation(x[3])
    'Block count: xx 00000: 1-4; 1+2=3'
    >>> x = bc.readTupleAndExtra(bs); x[1].explanation(x[3])
    'Block count: xxx 00110: 33-40; 33+0=33'
    >>> x = bc.readTupleAndExtra(bs); x[1].explanation(x[3])
    'Block count: xxxxxx 10001: 305-368; 305+28=333'
    >>> x = bc.readTupleAndExtra(bs); x[1].explanation(x[3])
    'Block count: xxxxxxxxxxx 10110: 2289-4336; 2289+1044=3333'
    """,

'Layout': """
    >>> olleke.pos = 0
    >>> l = Layout(olleke)
    >>> l.verboseRead(WindowSizeAlphabet())
    0000  1b                   1011 WSIZE   windowsize=(1<<22)-16=4194288
    4194288
    >>> l.verboseRead(BoolCode('LAST', description="Last block"))
                              1     LAST    Last block: 1: True
    True
    >>> l.verboseRead(BoolCode('EMPTY', description="Empty block"))
                             0      EMPTY   Empty block: 0: False
    False
    >>> l.verboseRead(MetablockLengthAlphabet())
    0001  2e 00        |00h|2Eh,|00 MLEN    data length: 002eh+1=47
    47
    >>> olleke.pos = 76
    >>> l = Layout(olleke)
    >>> x = l.verboseRead(DistanceAlphabet(0,NPOSTFIX=0,NDIRECT=0), skipExtra=True)
    000a  82                10|1100 D0      10[15*x]-3
    >>> x.explanation(0x86a3)
    '10[1000011010100011]-3: [0]+100000'
    """,

'olleke': """
    >>> olleke.pos = 0
    >>> try: Layout(olleke).processStream()
    ... except NotImplementedError: pass
    ... #doctest: +REPORT_NDIFF
    addr  hex               binary context explanation
    -----------------------Stream header------------------------
    0000  1b                   1011 WSIZE   windowsize=(1<<22)-16=4194288
    ======================Metablock header======================
                              1     LAST    Last block: 1: True
                             0      EMPTY   Empty block: 0: False
    0001  2e 00        |00h|2Eh,|00 MLEN    data length: 002eh+1=47
    -------------------Block type descriptors-------------------
    0003  00                      0 BT#L    0: 1 literal block type
                                 0  BT#I    0: 1 insert&copy block type
                                0   BT#D    0: 1 distance block type
    ------------------Distance code parameters------------------
    0004  44               0|000,00 DIST    0 postfix bits and 0000<<0=0 direct codes
    -----------------------Context modes------------------------
                         10         LC0     Context mode for type 0: 2(UTF8)
    ------------------------Context maps------------------------
                        0           LT#     0: 1 literal prefix tree
                       0            DT#     0: 1 distance prefix tree
    ---------------------Prefix code lists----------------------
                     10             PFX     L0 is complex with lengths 3,4,0,5,17...
    0005  4f                    1|0 ##L0    len 3: coded with 3 bits
                            0111    ##L0    len 4: coded with 1 bits
                          10        ##L0    unused: coded with 3 bits
    0006  d6                    0|0 ##L0    len 5: skipped
                             011    ##L0    zero xxx: coded with 2 bits
    ***** Lengths for L0 will be coded as:
      0:len 4     01:zero xxx 011:unused   111:len 3
    0007  95                1|11,01 #L0     7+3 unused
                           0        #L0     Length for \\n is 4 bits
                     001,01         #L0     1+3 unused
    0008  44                010,0|1 #L0     total 19+2 unused
                           0        #L0     Length for " " is 4 bits
                          0         #L0     Length for ! is 4 bits
    0009  cb                011,|01 #L0     3+3 unused
                     |110,01        #L0     total 35+6 unused
    000a  82                      0 #L0     Length for K is 4 bits
                            000,01  #L0     0+3 unused
                           0        #L0     Length for O is 4 bits
    000b  4d                   01|1 #L0     symbol P unused
                            011     #L0     symbol Q unused
                           0        #L0     Length for R is 4 bits
    000c  88                000,|01 #L0     0+3 unused
                     |100,01        #L0     total 11+4 unused
    000d  b6                      0 #L0     Length for b is 4 bits
                               011  #L0     symbol c unused
                            011     #L0     symbol d unused
    000e  27                   11|1 #L0     Length for e is 3 bits
                         010,01     #L0     2+3 unused
                       |0           #L0     Length for k is 4 bits
    000f  1f                    111 #L0     Length for l is 3 bits
                             011    #L0     symbol m unused
                            0       #L0     Length for n is 4 bits
                          |0        #L0     Length for o is 4 bits
    0010  c1                 000,01 #L0     0+3 unused
                            0       #L0     Length for s is 4 bits
    0011  b4                   0|11 #L0     symbol t unused
                              0     #L0     Length for u is 4 bits
    End of table. Prefix code L0:
     000:e   0010:\\n  0110:!   0001:O   0101:b   0011:n   0111:s
     100:l   1010:" " 1110:K   1001:R   1101:k   1011:o   1111:u
                         11,01      PFX     IC0 is simple with 4 code words
    0012  2a                |2Ah|10 IC0     ? bits: I5C4
    0013  b5 ec              00|B5h IC0     ? bits: I6+xC7
    0015  22            0010|111011 IC0     ? bits: I8+xC5
    0016  8c            001100|0010 IC0     ? bits: I0C14+xx
                       0            SHAPE   False: lengths 2,2,2,2
    0017  74                 10,0|1 PFX     D0 is simple with 3 code words
    0018  a6                0|01110 D0      1 bit: 2last-3
                      010011        D0      2 bits: 11xx-3
    0019  aa                01010|1 D0      2 bits: 11xxx-3
    ====================Meta block contents=====================
                       |1,01        IC0     Literal: 9, copy: 5
    001a  41                   0001 0,0=L0  O
                            100     0,48=L0 l
    001b  a2                   10|0 0,62=L0 l
                            000     0,63=L0 e
    001c  a1                  1|101 0,59=L0 k
                           000      0,63=L0 e
                      |1010         0,59=L0 " "
    001d  b5                   0101 0,11=L0 b
                          |1011     0,60=L0 o
    001e  24                      0 0,3=D0  Distance: 2last-3=8
                                            Seen before: "lleke"
                              0,10  IC0     Literal: 6, copy: 7
                         |0010      0,59=L0 \\n
    001f  89                   1001 0,7=L0  R
                            000     0,52=L0 e
    0020  fa                  010|1 0,58=L0 b
                          1111      0,63=L0 u
    0021  eb                  011|1 0,59=L0 s
                         11,01      0,3=D0  Absolute value: 12 (pos 8)
                                            Seen before: "olleke\\n"
    0022  db                 01,1|1 IC0     Literal: 0, copy: 15
                      |110,11       0,3=D0  Absolute value: 27 (pos 0)
                                            Seen before: "Olleke bolleke\\n"
    0023  f8                     00 IC0     Literal: 5, copy: 4
                             1110   0,7=L0  K
    0024  2c                  00|11 0,52=L0 n
                          1011      0,62=L0 o
    0025  0d                   1|00 0,59=L0 l
                           0110     0,63=L0 !
    """,

'file': """
    >>> try: Layout(BitStream(
    ... open("H:/Downloads/brotli-master/tests/testdata/10x10y.compressed",'rb')
    ...     .read())).processStream()
    ... except NotImplementedError: pass
    addr  hex               binary context explanation
    -----------------------Stream header------------------------
    0000  1b                   1011 WSIZE   windowsize=(1<<22)-16=4194288
    ======================Metablock header======================
                              1     LAST    Last block: 1: True
                             0      EMPTY   Empty block: 0: False
    0001  13 00        |00h|13h,|00 MLEN    data length: 0013h+1=20
    -------------------Block type descriptors-------------------
    0003  00                      0 BT#L    0: 1 literal block type
                                 0  BT#I    0: 1 insert&copy block type
                                0   BT#D    0: 1 distance block type
    ------------------Distance code parameters------------------
    0004  a4               0|000,00 DIST    0 postfix bits and 0000<<0=0 direct codes
    -----------------------Context modes------------------------
                         10         LC0     Context mode for type 0: 2(UTF8)
    ------------------------Context maps------------------------
                        0           LT#     0: 1 literal prefix tree
                       0            DT#     0: 1 distance prefix tree
    ---------------------Prefix code lists----------------------
    0005  b0                 0|1,01 PFX     L0 is simple with 2 code words
    0006  b2              0|1011000 L0      1 bit: X
    0007  ea              0|1011001 L0      1 bit: Y
                     01,01          PFX     IC0 is simple with 2 code words
    0008  81            0000001|111 IC0     1 bit: I1C9&D=0
    0009  47 02             0|47h|1 IC0     1 bit: I1C9
                       00,01        PFX     D0 is simple with 1 code word
    000b  8a                010|000 D0      0 bits: 10x-3
    ====================Meta block contents=====================
                           1        IC0     Literal: 1, copy: 9
                          0         0,0=L0  X
                      0,()          0,3=D0  Absolute value: 1 (pos 0)
                                            Seen before: "XXXXXXXXX"
                     0              IC0     Literal: 1, copy: 9, same distance
                   |1               0,54=L0 Y
                                            Seen before: "YYYYYYYYY"
    """,

'XY': """
    >>> try: Layout(BitStream(brotli.compress('X'*10+'Y'*10))).processStream()
    ... except NotImplementedError: pass
    addr  hex               binary context explanation
    -----------------------Stream header------------------------
    0000  1b                   1011 WSIZE   windowsize=(1<<22)-16=4194288
    ======================Metablock header======================
                              1     LAST    Last block: 1: True
                             0      EMPTY   Empty block: 0: False
    0001  13 00        |00h|13h,|00 MLEN    data length: 0013h+1=20
    -------------------Block type descriptors-------------------
    0003  00                      0 BT#L    0: 1 literal block type
                                 0  BT#I    0: 1 insert&copy block type
                                0   BT#D    0: 1 distance block type
    ------------------Distance code parameters------------------
    0004  a4               0|000,00 DIST    0 postfix bits and 0000<<0=0 direct codes
    -----------------------Context modes------------------------
                         10         LC0     Context mode for type 0: 2(UTF8)
    ------------------------Context maps------------------------
                        0           LT#     0: 1 literal prefix tree
                       0            DT#     0: 1 distance prefix tree
    ---------------------Prefix code lists----------------------
    0005  b0                 0|1,01 PFX     L0 is simple with 2 code words
    0006  b2              0|1011000 L0      1 bit: X
    0007  82              0|1011001 L0      1 bit: Y
                     00,01          PFX     IC0 is simple with 1 code word
    0008  84            0000100|100 IC0     0 bits: I4C6&D=0
    0009  00                 00,0|1 PFX     D0 is simple with 1 code word
    000a  e0                0|00000 D0      0 bits: last
    ====================Meta block contents=====================
                          ()        IC0     Literal: 4, copy: 6, same distance
                         0          0,0=L0  X
                        0           0,52=L0 X
                       0            0,54=L0 X
                      0             0,54=L0 X
                                            Seen before: "XXXXXX"
                    ()              IC0     Literal: 4, copy: 6, same distance
                   1                0,54=L0 Y
                  1                 0,54=L0 Y
                |1                  0,54=L0 Y
    000b  01                      1 0,54=L0 Y
                                            Seen before: "YYYYYY"
    """,

'empty': """
    >>> try: Layout(BitStream(b'\\x81\\x16\\x00\\x58')).processStream()
    ... except NotImplementedError: pass
    addr  hex               binary context explanation
    -----------------------Stream header------------------------
    0000  81                0000001 WSIZE   windowsize=(1<<17)-16=131056
    ======================Metablock header======================
                          |1        LAST    Last block: 1: True
    0001  16                      0 EMPTY   Empty block: 0: False
                                11  MLEN    11: empty block
                               0    RSVD    Reserved (must be zero)
    0002  00           000000|00,01 SKIP    skip length: 0h+1=1
                    |00             SKIP    2 bits ignored
    Skipping to 4
    """,

}

if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        l = Layout(BitStream(open(sys.argv[1],'rb').read()))
        l.processStream()
    else:
        sys.path.append("h:/Persoonlijk/bin")
        try:
            import brotli
            open('brotlidump.br', 'wb').write(
                brotli.compress(
                    open('brotlidump.py', 'r').read()
                ))
            olleke = BitStream(brotli.compress(
                'Olleke bolleke\nRebusolleke\nOlleke bolleke\nKnol!'))
        except ImportError: pass
        import doctest
        doctest.testmod(optionflags=doctest.REPORT_NDIFF
            #|doctest.FAIL_FAST
            )
