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

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class AbbreviationMapTest {
    private AbbreviationMap<String> abbreviations;

    @Before
    public void setUp() {
        abbreviations = new AbbreviationMap<>();
    }

    @Test( expected = NullPointerException.class )
    public void nullValuesAreIllegal() {
        abbreviations.put( "cannotAddNullValue", null );
    }

    @Test( expected = NullPointerException.class )
    public void nullKeysAreIllegalToAdd() {
        abbreviations.put( null, "-1" );
    }

    @Test( expected = NullPointerException.class )
    public void nullKeysAreIllegalToCheckForContains() {
        abbreviations.contains( null );
    }

    @Test( expected = IllegalArgumentException.class )
    public void zeroLengthKeysAreIllegalToAdd() {
        abbreviations.put( "", "1" );
    }

    @Test( expected = NullPointerException.class )
    public void nullKeysAreIllegalToRemove() {
        abbreviations.remove( null );
    }

    @Test( expected = IllegalArgumentException.class )
    public void zeroLengthKeysAreIllegalToRemove() {
        abbreviations.remove( "" );
    }

    @Test
    public void empty() {
        assertFalse( abbreviations.contains( "boo" ) );
        assertNull( abbreviations.get( "boo" ) );
    }

    @Test
    public void addingOne() {
        abbreviations.put( "box", "2" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "2", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "2", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "2", abbreviations.get( "b" ) );
    }

    @Test
    public void addingManyWithNoCommonPrefix() {
        abbreviations.put( "box", "1" );
        abbreviations.put( "cat", "2" );
        abbreviations.put( "dog", "3" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "1", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "1", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "1", abbreviations.get( "b" ) );
        assertTrue( abbreviations.contains( "cat" ) );
        assertEquals( "2", abbreviations.get( "cat" ) );
        assertTrue( abbreviations.contains( "ca" ) );
        assertEquals( "2", abbreviations.get( "ca" ) );
        assertTrue( abbreviations.contains( "c" ) );
        assertEquals( "2", abbreviations.get( "c" ) );
        assertTrue( abbreviations.contains( "dog" ) );
        assertEquals( "3", abbreviations.get( "dog" ) );
        assertTrue( abbreviations.contains( "do" ) );
        assertEquals( "3", abbreviations.get( "do" ) );
        assertTrue( abbreviations.contains( "d" ) );
        assertEquals( "3", abbreviations.get( "d" ) );
    }

    @Test
    public void addingTwoWithCommonPrefix() {
        abbreviations.put( "box", "3" );
        abbreviations.put( "boy", "4" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "3", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "boy" ) );
        assertEquals( "4", abbreviations.get( "boy" ) );
        assertFalse( abbreviations.contains( "bo" ) );
        assertNull( abbreviations.get( "bo" ) );
        assertFalse( abbreviations.contains( "b" ) );
        assertNull( abbreviations.get( "b" ) );
    }

    @Test
    public void addingThreeWithSuccessivelySmallerPrefixes() {
        abbreviations.put( "boy", "3" );
        abbreviations.put( "bo", "2" );
        abbreviations.put( "b", "1" );

        assertTrue( abbreviations.contains( "boy" ) );
        assertEquals( "3", abbreviations.get( "boy" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "2", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "1", abbreviations.get( "b" ) );
    }

    @Test
    public void addingThreeWithSuccessivelyLargerPrefixes() {
        abbreviations.put( "b", "1" );
        abbreviations.put( "bo", "2" );
        abbreviations.put( "boy", "3" );

        assertTrue( abbreviations.contains( "boy" ) );
        assertEquals( "3", abbreviations.get( "boy" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "2", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "1", abbreviations.get( "b" ) );
    }

    @Test
    public void addingThreeWithMixOfPrefixSize() {
        abbreviations.put( "boy", "3" );
        abbreviations.put( "b", "1" );
        abbreviations.put( "bo", "2" );

        assertTrue( abbreviations.contains( "boy" ) );
        assertEquals( "3", abbreviations.get( "boy" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "2", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "1", abbreviations.get( "b" ) );
    }

    @Test
    public void addingOneThenReplacing() {
        abbreviations.put( "box", "2" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "2", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "2", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "2", abbreviations.get( "b" ) );

        abbreviations.put( "box", "3" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "3", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "3", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "3", abbreviations.get( "b" ) );
    }

    @Test
    public void removingNonExistentKeyHasNoEffect() {
        abbreviations.put( "box", "2" );

        abbreviations.remove( "cat" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "2", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "2", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "2", abbreviations.get( "b" ) );
    }

    @Test
    public void removingSingleKey() {
        abbreviations.put( "box", "3" );

        abbreviations.remove( "box" );

        assertFalse( abbreviations.contains( "box" ) );
        assertNull( abbreviations.get( "box" ) );
        assertFalse( abbreviations.contains( "bo" ) );
        assertNull( abbreviations.get( "bo" ) );
        assertFalse( abbreviations.contains( "b" ) );
        assertNull( abbreviations.get( "b" ) );
    }

    @Test
    public void cannotRemoveByUniqueAbbreviation() {
        abbreviations.put( "box", "4" );

        abbreviations.remove( "bo" );
        abbreviations.remove( "b" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "4", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "4", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "4", abbreviations.get( "b" ) );
    }

    @Test
    public void removeKeyWithCommonPrefix() {
        abbreviations.put( "box", "-1" );
        abbreviations.put( "boy", "-2" );

        abbreviations.remove( "box" );

        assertFalse( abbreviations.contains( "box" ) );
        assertNull( abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "boy" ) );
        assertEquals( "-2", abbreviations.get( "boy" ) );
        assertTrue( abbreviations.contains( "bo" ) );
        assertEquals( "-2", abbreviations.get( "bo" ) );
        assertTrue( abbreviations.contains( "b" ) );
        assertEquals( "-2", abbreviations.get( "b" ) );
    }

    @Test
    public void addKeysWithCommonPrefixThenRemoveNonExistentKeyWithCommonPrefix() {
        abbreviations.put( "box", "-1" );
        abbreviations.put( "boy", "-2" );

        abbreviations.remove( "bop" );

        assertTrue( abbreviations.contains( "box" ) );
        assertEquals( "-1", abbreviations.get( "box" ) );
        assertTrue( abbreviations.contains( "boy" ) );
        assertEquals( "-2", abbreviations.get( "boy" ) );
        assertFalse( abbreviations.contains( "bo" ) );
        assertNull( abbreviations.get( "bo" ) );
        assertFalse( abbreviations.contains( "b" ) );
        assertNull( abbreviations.get( "b" ) );
    }

    @Test
    public void addKeysWithCommonPrefixesStairstepStyle() {
        abbreviations.put( "a", "1" );
        abbreviations.put( "abc", "2" );

        assertTrue( abbreviations.contains( "a" ) );
        assertEquals( "1", abbreviations.get( "a" ) );
        assertTrue( abbreviations.contains( "ab" ) );
        assertEquals( "2", abbreviations.get( "ab" ) );
        assertTrue( abbreviations.contains( "abc" ) );
        assertEquals( "2", abbreviations.get( "abc" ) );

        abbreviations.put( "abcde", "3" );

        assertTrue( abbreviations.contains( "a" ) );
        assertEquals( "1", abbreviations.get( "a" ) );
        assertFalse( abbreviations.contains( "ab" ) );
        assertNull( abbreviations.get( "ab" ) );
        assertTrue( abbreviations.contains( "abc" ) );
        assertEquals( "2", abbreviations.get( "abc" ) );
        assertTrue( abbreviations.contains( "abcd" ) );
        assertEquals( "3", abbreviations.get( "abcd" ) );
        assertTrue( abbreviations.contains( "abcde" ) );
        assertEquals( "3", abbreviations.get( "abcde" ) );
    }

    @Test
    public void addKeysWithCommonPrefixesStairstepStyleJumbled() {
        abbreviations.put( "a", "1" );
        abbreviations.put( "abcde", "3" );
        abbreviations.put( "abc", "2" );

        assertTrue( abbreviations.contains( "a" ) );
        assertEquals( "1", abbreviations.get( "a" ) );
        assertFalse( abbreviations.contains( "ab" ) );
        assertNull( abbreviations.get( "ab" ) );
        assertTrue( abbreviations.contains( "abc" ) );
        assertEquals( "2", abbreviations.get( "abc" ) );
        assertTrue( abbreviations.contains( "abcd" ) );
        assertEquals( "3", abbreviations.get( "abcd" ) );
        assertTrue( abbreviations.contains( "abcde" ) );
        assertEquals( "3", abbreviations.get( "abcde" ) );
    }

    @Test
    public void multipleKeysWithCommonPrefix() {
        abbreviations.put( "good", "4" );
        abbreviations.put( "goodyear", "8" );
        abbreviations.put( "go", "2" );
        abbreviations.put( "goodyea", "7" );
        abbreviations.put( "goodye", "6" );

        assertFalse( abbreviations.contains( "g" ) );
        assertNull( abbreviations.get( "g" ) );
        assertTrue( abbreviations.contains( "go" ) );
        assertEquals( "2", abbreviations.get( "go" ) );
        assertFalse( abbreviations.contains( "goo" ) );
        assertNull( abbreviations.get( "goo" ) );
        assertTrue( abbreviations.contains( "good" ) );
        assertEquals( "4", abbreviations.get( "good" ) );
        assertFalse( abbreviations.contains( "goody" ) );
        assertNull( abbreviations.get( "goody" ) );
        assertTrue( abbreviations.contains( "goodye" ) );
        assertEquals( "6", abbreviations.get( "goodye" ) );
        assertTrue( abbreviations.contains( "goodyea" ) );
        assertEquals( "7", abbreviations.get( "goodyea" ) );
        assertTrue( abbreviations.contains( "goodyea" ) );
        assertEquals( "8", abbreviations.get( "goodyear" ) );

        abbreviations.remove( "goodyea" );

        assertFalse( abbreviations.contains( "g" ) );
        assertNull( abbreviations.get( "g" ) );
        assertTrue( abbreviations.contains( "go" ) );
        assertEquals( "2", abbreviations.get( "go" ) );
        assertFalse( abbreviations.contains( "goo" ) );
        assertNull( abbreviations.get( "goo" ) );
        assertTrue( abbreviations.contains( "good" ) );
        assertEquals( "4", abbreviations.get( "good" ) );
        assertFalse( abbreviations.contains( "goody" ) );
        assertNull( abbreviations.get( "goody" ) );
        assertTrue( abbreviations.contains( "goodye" ) );
        assertEquals( "6", abbreviations.get( "goodye" ) );
        assertTrue( abbreviations.contains( "goodyea" ) );
        assertEquals( "8", abbreviations.get( "goodyea" ) );
        assertTrue( abbreviations.contains( "goodyea" ) );
        assertEquals( "8", abbreviations.get( "goodyear" ) );
    }
}
