package joptsimple.examples;

import static java.util.Arrays.*;

import joptsimple.OptionException;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.*;
import static org.junit.rules.ExpectedException.*;

public class ShortOptionsWithMultipleArgumentsForSingleOptionTest {
    @Rule public final ExpectedException thrown = none();

    @Test
    public void allowsMultipleValuesForAnOption() {
        OptionParser parser = new OptionParser( "a:" );

        OptionSet options = parser.parse( "-a", "foo", "-abar", "-a=baz" );

        assertTrue( options.has( "a" ) );
        assertTrue( options.hasArgument( "a" ) );
        assertEquals( asList( "foo", "bar", "baz" ), options.valuesOf( "a" ) );

        thrown.expect( OptionException.class );
        options.valueOf( "a" );
    }
}
