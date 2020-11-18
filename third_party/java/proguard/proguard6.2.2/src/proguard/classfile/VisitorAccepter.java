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
package proguard.classfile;




/**
 * This interface is a base interface for visitor accepters. It allows
 * visitors to set and get any temporary information they desire on the
 * objects they are visiting. Note that every visitor accepter has only one
 * such property, so visitors will have to take care not to overwrite each
 * other's information, if it is still required.
 *
 * @author Eric Lafortune
 */
public interface VisitorAccepter
{
    /**
     * Gets the visitor information of the visitor accepter.
     */
    public Object getVisitorInfo();


    /**
     * Sets the visitor information of the visitor accepter.
     */
    public void setVisitorInfo(Object visitorInfo);
}
