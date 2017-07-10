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

package joptsimple.internal;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static joptsimple.internal.Reflection.*;
import static org.junit.rules.ExpectedException.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ReflectionTest {
    @Rule public final ExpectedException thrown = none();

    @Test
    public void invokingConstructorQuietlyWrapsInstantiationException() throws Exception {
        Constructor<AbstractProblematic> constructor = AbstractProblematic.class.getDeclaredConstructor();

        thrown.expect( ReflectionException.class );
        thrown.expectMessage( InstantiationException.class.getName() );

        instantiate( constructor );
    }

    @Test
    public void invokingConstructorQuietlyWrapsIllegalAccessException() throws Exception {
        Constructor<Problematic> constructor = Problematic.class.getDeclaredConstructor();

        thrown.expect( ReflectionException.class );
        thrown.expectMessage( IllegalAccessException.class.getName() );

        instantiate( constructor );
    }

    @Test
    public void invokingConstructorQuietlyWrapsCauseOfInvocationTargetException() throws Exception {
        Constructor<Problematic> constructor = Problematic.class.getDeclaredConstructor( String.class );

        thrown.expect( ReflectionException.class );
        thrown.expectMessage( IllegalStateException.class.getName() );

        instantiate( constructor, "arg" );
    }

    @Test
    public void invokingConstructorQuietlyWrapsIllegalArgumentException() throws Exception {
        Constructor<Problematic> constructor = Problematic.class.getDeclaredConstructor(String.class);

        thrown.expect( ReflectionException.class );
        thrown.expectMessage( IllegalArgumentException.class.getName() );

        instantiate( constructor );
    }

    @Test
    public void invokingStaticMethodQuietlyWrapsIllegalAccessException() throws Exception {
        Method method = Problematic.class.getDeclaredMethod( "boo" );

        thrown.expect( ReflectionException.class );
        thrown.expectMessage( IllegalAccessException.class.getName() );

        invoke( method );
    }

    @Test
    public void invokingStaticMethodQuietlyWrapsIllegalArgumentException() throws Exception {
        Method method = Problematic.class.getDeclaredMethod( "mute" );

        thrown.expect( ReflectionException.class );
        thrown.expectMessage( IllegalArgumentException.class.getName() );

        invoke( method, new Object() );
    }

    private abstract static class AbstractProblematic {
        protected AbstractProblematic() {
            // no-op
        }
    }
}
