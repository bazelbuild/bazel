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
package proguard.optimize;

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;
import proguard.optimize.info.ReadWriteFieldMarker;

/**
 * This <code>MemberVisitor</code> delegates its visits to program fields to
 * other given <code>MemberVisitor</code> instances, but only when the visited
 * field has been marked as write-only.
 *
 * @see ReadWriteFieldMarker
 * @author Eric Lafortune
 */
public class WriteOnlyFieldFilter
extends      SimplifiedVisitor
implements   MemberVisitor
{
    private final MemberVisitor writeOnlyFieldVisitor;


    /**
     * Creates a new WriteOnlyFieldFilter.
     * @param writeOnlyFieldVisitor the <code>MemberVisitor</code> to which
     *                              visits to write-only fields will be delegated.
     */
    public WriteOnlyFieldFilter(MemberVisitor writeOnlyFieldVisitor)
    {
        this.writeOnlyFieldVisitor = writeOnlyFieldVisitor;
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {

        if (ReadWriteFieldMarker.isWritten(programField) &&
            !ReadWriteFieldMarker.isRead(programField))
        {
            writeOnlyFieldVisitor.visitProgramField(programClass, programField);
        }
    }
}
