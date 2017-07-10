package joptsimple.examples;

import joptsimple.OptionException;
import joptsimple.OptionParser;
import joptsimple.OptionSet;
import org.junit.Test;

import static org.junit.Assert.*;

public class RequiredOptionsTest {
    @Test( expected = OptionException.class )
    public void allowsSpecificationOfRequiredOptions() {
        OptionParser parser = new OptionParser() {
            {
                accepts( "userid" ).withRequiredArg().required();
                accepts( "password" ).withRequiredArg().required();
            }
        };

        parser.parse( "--userid", "bob" );
    }

    @Test
    public void aHelpOptionMeansRequiredOptionsNeedNotBePresent() {
        OptionParser parser = new OptionParser() {
            {
                accepts( "userid" ).withRequiredArg().required();
                accepts( "password" ).withRequiredArg().required();
                accepts( "help" ).forHelp();
            }
        };

        OptionSet options = parser.parse( "--help" );
        assertTrue( options.has( "help" ) );
    }
    
    @Test( expected = OptionException.class )
    public void missingHelpOptionMeansRequiredOptionsMustBePresent() {
        OptionParser parser = new OptionParser() {
            {
                accepts( "userid" ).withRequiredArg().required();
                accepts( "password" ).withRequiredArg().required();
                accepts( "help" ).forHelp();
            }
        };

        parser.parse( "" );
    }
}
