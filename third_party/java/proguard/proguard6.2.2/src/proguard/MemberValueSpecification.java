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
package proguard;

import proguard.util.ArrayUtil;

/**
 * This member specification assigns a constant value or value range to the
 * class members.
 *
 * @author Eric Lafortune
 */
public class MemberValueSpecification extends MemberSpecification
{
    public Number[] values;


    /**
     * Creates a new option to keep all possible class members.
     */
    public MemberValueSpecification()
    {
        this(0,
             0,
             null,
             null,
             null,
             null);
    }


    /**
     * Creates a new option to keep the specified class member(s).
     *
     * @param requiredSetAccessFlags   the class access flags that must be set
     *                                 in order for the class to apply.
     * @param requiredUnsetAccessFlags the class access flags that must be unset
     *                                 in order for the class to apply.
     * @param annotationType           the name of the class that must be an
     *                                 annotation in order for the class member
     *                                 to apply. The name may be null to specify
     *                                 that no annotation is required.
     * @param name                     the class member name. The name may be
     *                                 null to specify any class member or it
     *                                 may contain "*" or "?" wildcards.
     * @param descriptor               the class member descriptor. The
     *                                 descriptor may be null to specify any
     *                                 class member or it may contain
     *                                 "**", "*", or "?" wildcards.
     * @param values                   the constant value or value range
     *                                 assigned to this class member.
     */
    public MemberValueSpecification(int      requiredSetAccessFlags,
                                    int      requiredUnsetAccessFlags,
                                    String   annotationType,
                                    String   name,
                                    String   descriptor,
                                    Number[] values)
    {
        super(requiredSetAccessFlags,
              requiredUnsetAccessFlags,
              annotationType,
              name,
              descriptor);

        this.values = values;
    }



    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null)
        {
            return false;
        }

        MemberValueSpecification other = (MemberValueSpecification)object;
        return
            super.equals(other) &&
            ArrayUtil.equalOrNull(values, other.values);
    }

    public int hashCode()
    {
        return
            super.hashCode() ^
            ArrayUtil.hashCodeOrNull(values);
    }
}
