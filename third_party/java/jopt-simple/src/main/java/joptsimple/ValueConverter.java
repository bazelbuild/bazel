/*
 The MIT License

 Copyright (c) 2004-2015 Paul R. Holser, Jr.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

package joptsimple;

/**
 * Instances of this interface are used to convert arguments of options into specific Java types.
 *
 * @param <V> constraint on the type of values being converted to
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public interface ValueConverter<V> {
    /**
     * Converts the given string value into a Java type.
     *
     * @param value the string to convert
     * @return the converted value
     * @throws ValueConversionException if a problem occurs while converting the value
     */
    V convert( String value );

    /**
     * Gives the class of the type of values this converter converts to.
     *
     * @return the target class for conversion
     */
    Class<? extends V> valueType();

    /**
     * Gives a string that describes the pattern of the values this converter expects, if any.  For example, a date
     * converter can respond with a {@link java.text.SimpleDateFormat date format string}.
     *
     * @return a value pattern, or {@code null} if there's nothing interesting here
     */
    String valuePattern();
}
