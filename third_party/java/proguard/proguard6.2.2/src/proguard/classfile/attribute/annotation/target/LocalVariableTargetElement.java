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
package proguard.classfile.attribute.annotation.target;

/**
 * Representation of an local variable target table entry.
 *
 * @author Eric Lafortune
 */
public class LocalVariableTargetElement
{
    public int u2startPC;
    public int u2length;
    public int u2index;

    /**
     * Creates an uninitialized LocalVariableTargetElement.
     */
    public LocalVariableTargetElement()
    {
    }


    /**
     * Creates an initialized LocalVariableTargetElement.
     */
    public LocalVariableTargetElement(int u2startPC,
                                      int u2length,
                                      int u2index)
    {
        this.u2startPC = u2startPC;
        this.u2length  = u2length;
        this.u2index   = u2index;
    }
}
