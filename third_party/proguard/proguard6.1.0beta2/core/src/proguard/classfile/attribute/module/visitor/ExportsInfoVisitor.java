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
package proguard.classfile.attribute.module.visitor;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.module.ExportsInfo;

/**
 * This interface specifies the methods for a visitor of
 * <code>ExportsInfo</code> objects. Note that there is only a single
 * implementation of <code>ExportsInfo</code>, such that this interface
 * is not strictly necessary as a visitor.
 *
 * @author Joachim Vandersmissen
 */
public interface ExportsInfoVisitor
{
    public void visitExportsInfo(Clazz clazz, ExportsInfo exportsInfo);
}
