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

import java.util.List;

/**
 * Describes options that an option parser recognizes.
 *
 * <p>Instances of this interface are returned by the "fluent interface" methods to allow retrieval of option arguments
 * in a type-safe manner.  Here's an example:</p>
 * 
 * <pre><code>
 *     OptionParser parser = new OptionParser();
 *     <strong>OptionSpec&lt;Integer&gt;</strong> count =
 *         parser.accepts( "count" ).withRequiredArg().ofType( Integer.class );
 *     OptionSet options = parser.parse( "--count", "2" );
 *     assert options.has( count );
 *     int countValue = options.valueOf( count );
 *     assert countValue == count.value( options );
 *     List&lt;Integer&gt; countValues = options.valuesOf( count );
 *     assert countValues.equals( count.values( options ) );
 * </code></pre>
 *
 * @param <V> represents the type of the arguments this option accepts
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public interface OptionSpec<V> {
    /**
     * Gives any arguments associated with the given option in the given set of detected options.
     *
     * <p>Specifying a {@linkplain ArgumentAcceptingOptionSpec#defaultsTo(Object, Object[]) default argument value}
     * for this option will cause this method to return that default value even if this option was not detected on the
     * command line, or if this option can take an optional argument but did not have one on the command line.</p>
     *
     * @param detectedOptions the detected options to search in
     * @return the arguments associated with this option; an empty list if no such arguments are present, or if this
     * option was not detected
     * @throws OptionException if there is a problem converting this option's arguments to the desired type; for
     * example, if the type does not implement a correct conversion constructor or method
     * @throws NullPointerException if {@code detectedOptions} is {@code null}
     * @see OptionSet#valuesOf(OptionSpec)
     */
    List<V> values( OptionSet detectedOptions );

    /**
     * Gives the argument associated with the given option in the given set of detected options.
     *
     * <p>Specifying a {@linkplain ArgumentAcceptingOptionSpec#defaultsTo(Object, Object[]) default argument value}
     * for this option will cause this method to return that default value even if this option was not detected on the
     * command line, or if this option can take an optional argument but did not have one on the command line.</p>
     *
     * @param detectedOptions the detected options to search in
     * @return the argument of the this option; {@code null} if no argument is present, or that option was not detected
     * @throws OptionException if more than one argument was detected for the option
     * @throws NullPointerException if {@code detectedOptions} is {@code null}
     * @throws ClassCastException if the arguments of this option are not of the expected type
     * @see OptionSet#valueOf(OptionSpec)
     */
    V value( OptionSet detectedOptions );

    /**
     * @return the string representations of this option
     */
    List<String> options();

    /**
     * Tells whether this option is designated as a "help" option. The presence of a "help" option on a command line
     * means that missing "required" options will not cause parsing to fail.
     *
     * @return whether this option is designated as a "help" option
     */
    boolean isForHelp();
}
