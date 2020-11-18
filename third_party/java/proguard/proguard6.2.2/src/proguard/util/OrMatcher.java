/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.util;

/**
 * This StringMatcher tests whether strings matches either of the given
 *  StringMatcher instances.
 *
 * @author Eric Lafortune
 */
public class OrMatcher extends StringMatcher
{
    private final StringMatcher matcher1;
    private final StringMatcher matcher2;


    /**
     * Creates a new OrMatcher with the two given string matchers.
     */
    public OrMatcher(StringMatcher matcher1, StringMatcher matcher2)
    {
        this.matcher1 = matcher1;
        this.matcher2 = matcher2;
    }


    // Implementations for StringMatcher.

    @Override
    protected boolean matches(String string, int beginOffset, int endOffset)
    {
        return matcher1.matches(string, beginOffset, endOffset) ||
               matcher2.matches(string, beginOffset, endOffset);
    }
}
