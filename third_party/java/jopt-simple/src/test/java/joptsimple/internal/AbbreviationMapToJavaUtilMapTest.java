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

import static java.util.Collections.*;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class AbbreviationMapToJavaUtilMapTest {
    private AbbreviationMap<String> abbreviations;

    @Before
    public void setUp() {
        abbreviations = new AbbreviationMap<>();
    }

    @Test
    public void empty() {
        assertEquals( emptyMap(), abbreviations.toJavaUtilMap() );
    }

    @Test
    public void addingOne() {
        abbreviations.put( "box", "2" );

        assertEquals( singletonMap( "box", "2" ), abbreviations.toJavaUtilMap() );
    }

    @Test
    public void addingManyWithNoCommonPrefix() {
        abbreviations.put( "box", "1" );
        abbreviations.put( "cat", "2" );
        abbreviations.put( "dog", "3" );

        assertEquals( "{box=1, cat=2, dog=3}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void addingTwoWithCommonPrefix() {
        abbreviations.put( "box", "3" );
        abbreviations.put( "boy", "4" );

        assertEquals( "{box=3, boy=4}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void addingThreeWithSuccessivelySmallerPrefixes() {
        abbreviations.put( "boy", "3" );
        abbreviations.put( "bo", "2" );
        abbreviations.put( "b", "1" );

        assertEquals( "{b=1, bo=2, boy=3}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void addingThreeWithSuccessivelyLargerPrefixes() {
        abbreviations.put( "b", "1" );
        abbreviations.put( "bo", "2" );
        abbreviations.put( "boy", "3" );

        assertEquals( "{b=1, bo=2, boy=3}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void addingThreeWithMixOfPrefixSize() {
        abbreviations.put( "boy", "3" );
        abbreviations.put( "b", "1" );
        abbreviations.put( "bo", "2" );

        assertEquals( "{b=1, bo=2, boy=3}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void addingOneThenReplacing() {
        abbreviations.put( "box", "2" );
        abbreviations.put( "box", "3" );

        assertEquals( "{box=3}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void removeKeyWithCommonPrefix() {
        abbreviations.put( "box", "-1" );
        abbreviations.put( "boy", "-2" );
        abbreviations.remove( "box" );

        assertEquals( "{boy=-2}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void addKeysWithCommonPrefixesStairstepStyle() {
        abbreviations.put( "a", "1" );
        abbreviations.put( "abc", "2" );
        abbreviations.put( "abcde", "3" );

        assertEquals( "{a=1, abc=2, abcde=3}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void addKeysWithCommonPrefixesStairstepStyleJumbled() {
        abbreviations.put( "a", "1" );
        abbreviations.put( "abcde", "3" );
        abbreviations.put( "abc", "2" );

        assertEquals( "{a=1, abc=2, abcde=3}", abbreviations.toJavaUtilMap().toString() );
    }

    @Test
    public void multipleKeysWithCommonPrefix() {
        abbreviations.put( "good", "4" );
        abbreviations.put( "goodyear", "8" );
        abbreviations.put( "go", "2" );
        abbreviations.put( "goodyea", "7" );
        abbreviations.put( "goodye", "6" );
        abbreviations.remove( "goodyea" );

        assertEquals( "{go=2, good=4, goodye=6, goodyear=8}", abbreviations.toJavaUtilMap().toString() );
    }
}
