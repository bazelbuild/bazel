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

import java.util.Collections;
import java.util.List;

import org.infinitest.toolkit.UnmodifiableListTestSupport;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class OptionSetValuesOfOptionStringImmutabilityTest extends UnmodifiableListTestSupport<Object> {
    @SuppressWarnings( "unchecked" )
    @Override
    protected List<Object> newList() {
        RequiredArgumentOptionSpec<String> optionB = new RequiredArgumentOptionSpec<>( "b" );
        OptionSet options = new OptionSet( Collections.<String, AbstractOptionSpec<?>> emptyMap() );
        options.addWithArgument( optionB, "foo" ); 
        options.addWithArgument( optionB, "bar" );

        return (List<Object>) options.valuesOf( "b" );
    }

    @Override
    protected String newItem() {
        return "baz";
    }

    @Override
    protected String containedItem() {
        return "bar";
    }
}
