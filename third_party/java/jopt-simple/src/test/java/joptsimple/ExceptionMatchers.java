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

import java.lang.reflect.InvocationTargetException;

import org.hamcrest.Description;
import org.hamcrest.Matcher;
import org.hamcrest.TypeSafeMatcher;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ExceptionMatchers {
    private ExceptionMatchers() {
        throw new UnsupportedOperationException();
    }

    public static Matcher<OptionException> withOption( final String option ) {
        return new TypeSafeMatcher<OptionException>() {
            @Override
            public boolean matchesSafely( OptionException target ) {
                return target.options().contains( option );
            }

            public void describeTo( Description description ) {
                description.appendText( "an OptionException indicating the option ");
                description.appendValue( option );
            }
        };
    }

    public static Matcher<Throwable> withCauseOfType( final Class<? extends Throwable> type ) {
        return new TypeSafeMatcher<Throwable>() {
            @Override
            public boolean matchesSafely( Throwable target ) {
                return type.isInstance( target.getCause() );
            }

            public void describeTo( Description description ) {
                description.appendText( "an exception with cause of type " );
                description.appendValue( type );
            }
        };
    }

    public static Matcher<InvocationTargetException> withTargetOfType( final Class<? extends Throwable> type ) {
        return new TypeSafeMatcher<InvocationTargetException>() {
            @Override
            public boolean matchesSafely( InvocationTargetException target ) {
                return type.isInstance( target.getTargetException() );
            }

            public void describeTo( Description description ) {
                description.appendText( "an InvocationTargetException with target of type " );
                description.appendValue( type );
            }
        };
    }
}
