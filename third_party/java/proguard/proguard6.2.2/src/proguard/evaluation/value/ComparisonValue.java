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
 * This IntegerValue represents the result of a comparisons of two scalar
 * values.
 *
 * @author Eric Lafortune
 */
final class ComparisonValue extends SpecificIntegerValue
{
    private final Value value1;
    private final Value value2;


    /**
     * Creates a new comparison integer value of the two given scalar values.
     */
    public ComparisonValue(Value value1,
                           Value value2)
    {
        this.value1 = value1;
        this.value2 = value2;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        return this == object ||
               super.equals(object) &&
               this.value1.equals(((ComparisonValue)object).value1) &&
               this.value2.equals(((ComparisonValue)object).value2);
    }


    public int hashCode()
    {
        return super.hashCode() ^
               value1.hashCode() ^
               value2.hashCode();
    }


    public String toString()
    {
        return "("+value1+"~"+ value2 +")";
    }
}