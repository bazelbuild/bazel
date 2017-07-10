package joptsimple.examples;

import java.io.File;

import static java.io.File.*;
import static java.util.Arrays.*;

import joptsimple.OptionParser;

import static joptsimple.util.DateConverter.*;

public class HelpScreenExample {
    public static void main( String[] args ) throws Exception {
        OptionParser parser = new OptionParser() {
            {
                accepts( "c" ).withRequiredArg().ofType( Integer.class )
                    .describedAs( "count" ).defaultsTo( 1 );
                accepts( "q" ).withOptionalArg().ofType( Double.class )
                    .describedAs( "quantity" );
                accepts( "d", "some date" ).withRequiredArg().required()
                    .withValuesConvertedBy( datePattern( "MM/dd/yy" ) );
                acceptsAll( asList( "v", "talkative", "chatty" ), "be more verbose" );
                accepts( "output-file" ).withOptionalArg().ofType( File.class )
                     .describedAs( "file" );
                acceptsAll( asList( "h", "?" ), "show help" ).forHelp();
                acceptsAll( asList( "cp", "classpath" ) ).withRequiredArg()
                    .describedAs( "path1" + pathSeparatorChar + "path2:..." )
                    .ofType( File.class )
                    .withValuesSeparatedBy( pathSeparatorChar );
            }
        };

        parser.printHelpOn( System.out );
    }
}
