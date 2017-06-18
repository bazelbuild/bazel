package joptsimple.examples;

import joptsimple.OptionParser;

public class ExceptionExample {
    public static void main( String[] args ) {
        OptionParser parser = new OptionParser();

        parser.parse( "-x" );
    }
}
