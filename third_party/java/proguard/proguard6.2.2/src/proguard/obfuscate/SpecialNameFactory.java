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
package proguard.obfuscate;

/**
 * This <code>NameFactory</code> generates names that are special, by appending
 * a suffix.
 *
 * @author Eric Lafortune
 */
public class SpecialNameFactory implements NameFactory
{
    private static final char SPECIAL_SUFFIX = '_';


    private final NameFactory nameFactory;


    /**
     * Creates a new <code>SpecialNameFactory</code>.
     * @param nameFactory the name factory from which original names will be
     *                    retrieved.
     */
    public SpecialNameFactory(NameFactory nameFactory)
    {
        this.nameFactory = nameFactory;
    }


    // Implementations for NameFactory.

    public void reset()
    {
        nameFactory.reset();
    }


    public String nextName()
    {
        return nameFactory.nextName() + SPECIAL_SUFFIX;
    }


    // Small utility methods.

    /**
     * Returns whether the given name is special.
     */
    static boolean isSpecialName(String name)
    {
        return name != null &&
               name.charAt(name.length()-1) == SPECIAL_SUFFIX;
    }


    public static void main(String[] args)
    {
        SpecialNameFactory factory = new SpecialNameFactory(new SimpleNameFactory());

        for (int counter = 0; counter < 50; counter++)
        {
            System.out.println("["+factory.nextName()+"]");
        }
    }
}
