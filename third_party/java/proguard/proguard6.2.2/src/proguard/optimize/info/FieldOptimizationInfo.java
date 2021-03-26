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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.evaluation.value.*;

/**
 * This class stores some optimization information that can be attached to
 * a field.
 *
 * @author Eric Lafortune
 */
public class FieldOptimizationInfo
extends      SimplifiedVisitor
{
    protected Value value;


    public boolean isKept()
    {
        return true;
    }


    public boolean isWritten()
    {
        return true;
    }


    public boolean isRead()
    {
        return true;
    }


    public boolean canBeMadePrivate()
    {
        return false;
    }


    public ReferenceValue getReferencedClass()
    {
        return null;
    }


    public void setValue(Value value)
    {
        this.value = value;
    }


    public Value getValue()
    {
        return value;
    }


    public static void setFieldOptimizationInfo(Clazz clazz, Field field)
    {
        field.setVisitorInfo(new FieldOptimizationInfo());
    }


    public static FieldOptimizationInfo getFieldOptimizationInfo(Field field)
    {
        return (FieldOptimizationInfo)field.getVisitorInfo();
    }
}
