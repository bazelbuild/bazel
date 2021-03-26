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
 * NameFactory that prepends the names of the wrapped NameFactory with
 * a fixed prefix.
 *
 * @author Johan Leys
 */
public class PrefixingNameFactory implements NameFactory
{
    private final NameFactory delegateNameFactory;
    private final String      prefix;


    /**
     * Creates a new PrefixingNameFactory.
     * @param delegateNameFactory the wrapped NameFactory.
     * @param prefix              the prefix to add to all generated names.
     */
    public PrefixingNameFactory(NameFactory delegateNameFactory,
                                String      prefix)
    {
        this.delegateNameFactory = delegateNameFactory;
        this.prefix              = prefix;
    }


    // Implementations for NameFactory.

    public String nextName()
    {
        return prefix + delegateNameFactory.nextName();
    }

    public void reset()
    {
        delegateNameFactory.reset();
    }
}
