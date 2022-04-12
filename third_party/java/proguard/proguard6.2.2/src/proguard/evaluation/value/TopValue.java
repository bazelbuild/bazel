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
package proguard.evaluation.value;

/**
 * This class represents a partially evaluated top value. A top value is the
 * dummy value that takes up the extra space when storing a long value or a
 * double value.
 *
 * @author Eric Lafortune
 */
public class TopValue extends Category1Value
{
    // Implementations for Value.

    public boolean isSpecific()
    {
        return true;
    }

    public boolean isParticular()
    {
        return true;
    }

    public final Value generalize(Value other)
    {
        return this.getClass() == other.getClass() ? this : null;
    }

    public final int computationalType()
    {
        return TYPE_TOP;
    }

    public final String internalType()
    {
        return null;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return object != null &&
               this.getClass() == object.getClass();
    }


    public int hashCode()
    {
        return this.getClass().hashCode();
    }


    public String toString()
    {
        return "T";
    }
}
