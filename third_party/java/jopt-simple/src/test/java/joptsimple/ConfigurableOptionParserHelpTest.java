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

import java.io.StringWriter;
import java.math.BigDecimal;
import java.util.Date;
import java.util.List;

import static java.math.BigDecimal.*;
import static java.util.Arrays.*;
import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static joptsimple.internal.Strings.*;
import static joptsimple.util.DateConverter.*;
import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class ConfigurableOptionParserHelpTest extends AbstractOptionParserFixture {
    private StringWriter sink;

    @Before
    public final void createSink() {
        parser.formatHelpWith( new BuiltinHelpFormatter( 120, 3 ) );
        sink = new StringWriter();
    }

    @Test
    public void unconfiguredParser() throws Exception {
        parser.printHelpOn( sink );

        assertHelpLines( "No options specified   ", EMPTY );
    }

    @Test
    public void oneOptionNoArgNoDescription() throws Exception {
        parser.accepts( "apple" );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option    Description",
            "------    -----------",
            "--apple              ",
            EMPTY );
    }

    @Test
    public void oneOptionNoArgWithDescription() throws Exception {
        parser.accepts( "a", "some description" );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option   Description     ",
            "------   -----------     ",
            "-a       some description",
            EMPTY );
    }

    @Test
    public void twoOptionsNoArgWithDescription() throws Exception {
        parser.accepts( "a", "some description" );
        parser.accepts( "verbose", "even more description" );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option      Description          ",
            "------      -----------          ",
            "-a          some description     ",
            "--verbose   even more description",
            EMPTY );
    }

    @Test
    public void oneOptionRequiredArgNoDescription() throws Exception {
        parser.accepts( "a" ).withRequiredArg();

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option        Description",
            "------        -----------",
            "-a <String>              ",
            EMPTY );
    }

    @Test
    public void oneOptionRequiredArgNoDescriptionWithType() throws Exception {
        parser.accepts( "a" ).withRequiredArg().ofType( Integer.class );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option         Description",
            "------         -----------",
            "-a <Integer>              ",
            EMPTY );
    }

    @Test
    public void oneOptionRequiredArgWithDescription() throws Exception {
        parser.accepts( "a", "some value you need" ).withRequiredArg().describedAs( "numerical" );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                   Description        ",
            "------                   -----------        ",
            "-a <String: numerical>   some value you need",
            EMPTY );
    }

    @Test
    public void oneOptionRequiredArgWithDescriptionAndType() throws Exception {
        parser.accepts( "a", "some value you need" ).withRequiredArg().describedAs( "numerical" )
            .ofType( Integer.class );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                    Description        ",
            "------                    -----------        ",
            "-a <Integer: numerical>   some value you need",
            EMPTY );
    }

    @Test
    public void oneOptionOptionalArgNoDescription() throws Exception {
        parser.accepts( "threshold" ).withOptionalArg();

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                 Description",
            "------                 -----------",
            "--threshold [String]              ",
            EMPTY );
    }

    @Test
    public void oneOptionOptionalArgNoDescriptionWithType() throws Exception {
        parser.accepts( "a" ).withOptionalArg().ofType( Float.class );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option       Description",
            "------       -----------",
            "-a [Float]              ",
            EMPTY );
    }

    @Test
    public void oneOptionOptionalArgWithDescription() throws Exception {
        parser.accepts( "threshold", "some value you need" ).withOptionalArg().describedAs( "positive integer" );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                                   Description        ",
            "------                                   -----------        ",
            "--threshold [String: positive integer]   some value you need",
            EMPTY );
    }

    @Test
    public void oneOptionOptionalArgWithDescriptionAndType() throws Exception {
        parser.accepts( "threshold", "some value you need" ).withOptionalArg().describedAs( "positive decimal" )
            .ofType( Double.class );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                                   Description        ",
            "------                                   -----------        ",
            "--threshold [Double: positive decimal]   some value you need",
            EMPTY );
    }

    @Test
    public void alternativeLongOptions() throws Exception {
        parser.recognizeAlternativeLongOptions( true );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                   Description                     ",
            "------                   -----------                     ",
            "-W <String: opt=value>   Alternative form of long options",
            EMPTY );
    }

    @Test
    public void optionSynonymsWithoutArguments() throws Exception {
        parser.acceptsAll( asList( "v", "chatty" ), "be verbose" );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option         Description",
            "------         -----------",
            "-v, --chatty   be verbose ",
            EMPTY );
    }

    @Test
    public void optionSynonymsWithRequiredArgument() throws Exception {
        parser.acceptsAll( asList( "L", "index" ), "set level" ).withRequiredArg().ofType( Integer.class );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                  Description",
            "------                  -----------",
            "-L, --index <Integer>   set level  ",
            EMPTY );
    }

    @Test
    public void optionSynonymsWithOptionalArgument() throws Exception {
        parser.acceptsAll( asList( "d", "since" ), "date filter" ).withOptionalArg().describedAs( "yyyyMMdd" )
            .ofType( Date.class );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                         Description",
            "------                         -----------",
            "-d, --since [Date: yyyyMMdd]   date filter",
            EMPTY );
    }

    @Test
    public void optionSynonymsSortedByShortOptionThenLexicographical() throws Exception {
        parser.acceptsAll( asList( "v", "prolix", "chatty" ) );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                   Description",
            "------                   -----------",
            "-v, --chatty, --prolix              ",
            EMPTY );
    }

    @Test
    public void bothColumnsExceedingAllocatedWidths() throws Exception {
        parser.acceptsAll( asList( "t", "threshold", "cutoff" ),
                "a threshold value beyond which a certain level of the application should cease to write logs" )
                .withRequiredArg()
                .describedAs( "a positive decimal number that will represent the threshold that has been outlined" )
                .ofType( Double.class );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                                                     Description                                          ",
            "------                                                     -----------                                          ",
            "-t, --cutoff, --threshold <Double: a positive decimal      a threshold value beyond which a certain level of the",
            "  number that will represent the threshold that has been     application should cease to write logs             ",
            "  outlined>                                                                                                     ",
            EMPTY );
    }

    // Bug 2018262
    @Test
    public void gradleHelp() throws Exception {
        parser.acceptsAll( asList( "n", "non-recursive" ), "Do not execute primary tasks of child projects." );
        parser.acceptsAll( singletonList( "S" ),
            "Don't trigger a System.exit(0) for normal termination. Used for Gradle's internal testing." );
        parser.acceptsAll( asList( "I", "no-imports" ), "Disable usage of default imports for build script files." );
        parser.acceptsAll( asList( "u", "no-search-upward" ),
            "Don't search in parent folders for a settings.gradle file." );
        parser.acceptsAll( asList( "x", "cache-off" ), "No caching of compiled build scripts." );
        parser.acceptsAll( asList( "r", "rebuild-cache" ), "Rebuild the cache of compiled build scripts." );
        parser.acceptsAll( asList( "v", "version" ), "Print version info." );
        parser.acceptsAll( asList( "d", "debug" ), "Log in debug mode (includes normal stacktrace)." );
        parser.acceptsAll( asList( "q", "quiet" ), "Log errors only." );
        parser.acceptsAll( asList( "j", "ivy-debug" ), "Set Ivy log level to debug (very verbose)." );
        parser.acceptsAll( asList( "i", "ivy-quiet" ), "Set Ivy log level to quiet." );
        parser.acceptsAll( asList( "s", "stacktrace" ),
            "Print out the stacktrace also for user exceptions (e.g. compile error)." );
        parser.acceptsAll( asList( "f", "full-stacktrace" ),
            "Print out the full (very verbose) stacktrace for any exceptions." );
        parser.acceptsAll( asList( "t", "tasks" ), "Show list of all available tasks and their dependencies." );
        parser.acceptsAll( asList( "p", "project-dir" ),
            "Specifies the start dir for Gradle. Defaults to current dir." ).withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "g", "gradle-user-home" ), "Specifies the gradle user home dir." )
            .withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "l", "plugin-properties-file" ), "Specifies the plugin.properties file." )
            .withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "b", "buildfile" ),
            "Specifies the build file name (also for subprojects). Defaults to build.gradle." )
            .withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "D", "systemprop" ), "Set system property of the JVM (e.g. -Dmyprop=myvalue)." )
            .withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "P", "projectprop" ),
            "Set project property for the build script (e.g. -Pmyprop=myvalue)." )
            .withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "e", "embedded" ), "Specify an embedded build script." )
            .withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "B", "bootstrap-debug" ),
            "Specify a text to be logged at the beginning (e.g. used by Gradle's bootstrap class)." )
            .withRequiredArg().ofType( String.class );
        parser.acceptsAll( asList( "h", "?" ), "Shows this help message." ).forHelp();

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                                  Description                                                                    ",
            "------                                  -----------                                                                    ",
            "-?, -h                                  Shows this help message.                                                       ",
            "-B, --bootstrap-debug <String>          Specify a text to be logged at the beginning (e.g. used by Gradle's bootstrap  ",
            "                                          class).                                                                      ",
            "-D, --systemprop <String>               Set system property of the JVM (e.g. -Dmyprop=myvalue).                        ",
            "-I, --no-imports                        Disable usage of default imports for build script files.                       ",
            "-P, --projectprop <String>              Set project property for the build script (e.g. -Pmyprop=myvalue).             ",
            "-S                                      Don't trigger a System.exit(0) for normal termination. Used for Gradle's       ",
            "                                          internal testing.                                                            ",
            "-b, --buildfile <String>                Specifies the build file name (also for subprojects). Defaults to build.gradle.",
            "-d, --debug                             Log in debug mode (includes normal stacktrace).                                ",
            "-e, --embedded <String>                 Specify an embedded build script.                                              ",
            "-f, --full-stacktrace                   Print out the full (very verbose) stacktrace for any exceptions.               ",
            "-g, --gradle-user-home <String>         Specifies the gradle user home dir.                                            ",
            "-i, --ivy-quiet                         Set Ivy log level to quiet.                                                    ",
            "-j, --ivy-debug                         Set Ivy log level to debug (very verbose).                                     ",
            "-l, --plugin-properties-file <String>   Specifies the plugin.properties file.                                          ",
            "-n, --non-recursive                     Do not execute primary tasks of child projects.                                ",
            "-p, --project-dir <String>              Specifies the start dir for Gradle. Defaults to current dir.                   ",
            "-q, --quiet                             Log errors only.                                                               ",
            "-r, --rebuild-cache                     Rebuild the cache of compiled build scripts.                                   ",
            "-s, --stacktrace                        Print out the stacktrace also for user exceptions (e.g. compile error).        ",
            "-t, --tasks                             Show list of all available tasks and their dependencies.                       ",
            "-u, --no-search-upward                  Don't search in parent folders for a settings.gradle file.                     ",
            "-v, --version                           Print version info.                                                            ",
            "-x, --cache-off                         No caching of compiled build scripts.                                          ",
            EMPTY );
    }

    @Test
    public void dateConverterShowsDatePattern() throws Exception {
        parser.accepts( "date", "a date" ).withRequiredArg().withValuesConvertedBy( datePattern( "MM/dd/yy" ) );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option              Description",
            "------              -----------",
            "--date <MM/dd/yy>   a date     ",
            EMPTY );
    }

    @Test
    public void dateConverterShowsDatePatternInCombinationWithDescription() throws Exception {
        parser.accepts( "date", "a date" ).withOptionalArg().describedAs( "your basic date pattern" )
            .withValuesConvertedBy( datePattern( "MM/dd/yy" ) );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                                       Description",
            "------                                       -----------",
            "--date [MM/dd/yy: your basic date pattern]   a date     ",
            EMPTY );
    }

    @Test
    public void leavesEmbeddedNewlinesInDescriptionsAlone() throws Exception {
        List<String> descriptionPieces =
            asList( "Specify the output type.", "'raw' = raw data.", "'java' = java class" );
        parser.accepts( "type", join( descriptionPieces, LINE_SEPARATOR ) );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option   Description             ",
            "------   -----------             ",
            "--type   Specify the output type.",
            "         'raw' = raw data.       ",
            "         'java' = java class     ",
            EMPTY );
    }

    @Test
    public void includesDefaultValueForRequiredOptionArgument() throws Exception {
        parser.accepts( "a" ).withRequiredArg().defaultsTo( "boo" );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option        Description   ",
            "------        -----------   ",
            "-a <String>   (default: boo)",
            EMPTY );
    }

    @Test
    public void includesDefaultValueForOptionalOptionArgument() throws Exception {
        parser.accepts( "b" ).withOptionalArg().ofType( Integer.class ).defaultsTo( 5 );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option         Description ",
            "------         ----------- ",
            "-b [Integer]   (default: 5)",
            EMPTY );
    }

    @Test
    public void includesDefaultValueForArgumentWithDescription() throws Exception {
        parser.accepts( "c", "a quantity" ).withOptionalArg().ofType( BigDecimal.class )
            .describedAs( "quantity" ).defaultsTo( TEN );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                      Description             ",
            "------                      -----------             ",
            "-c [BigDecimal: quantity]   a quantity (default: 10)",
            EMPTY );
    }

    @Test
    public void includesListOfDefaultsForArgumentWithDescription() throws Exception {
        parser.accepts( "d", "dizzle" ).withOptionalArg().ofType( Integer.class )
            .describedAs( "double dizzle" ).defaultsTo( 2, 3, 5, 7 );

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option                        Description                   ",
            "------                        -----------                   ",
            "-d [Integer: double dizzle]   dizzle (default: [2, 3, 5, 7])",
            EMPTY );
    }

    @Test
    public void marksRequiredOptionsSpecially() throws Exception {
        parser.accepts( "e" ).withRequiredArg().required();

        parser.printHelpOn( sink );

        assertHelpLines(
            "Option (* = required)   Description",
            "---------------------   -----------",
            "* -e <String>                      ",
            EMPTY );
    }

    private void assertHelpLines( String... expectedLines ) {
        assertEquals( join( expectedLines, LINE_SEPARATOR ), sink.toString() );
    }
}
