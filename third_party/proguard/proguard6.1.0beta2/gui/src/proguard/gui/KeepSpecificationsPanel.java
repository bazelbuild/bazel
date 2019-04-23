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
package proguard.gui;

import proguard.*;

import javax.swing.*;

/**
 * This <code>ListPanel</code> allows the user to add, edit, move, and remove
 * KeepClassSpecification entries in a list.
 *
 * @author Eric Lafortune
 */
final class KeepSpecificationsPanel extends ClassSpecificationsPanel
{
    private final boolean markClasses;
    private final boolean markConditionally;
    private final boolean markDescriptorClasses;
    private final boolean allowShrinking;
    private final boolean allowOptimization;
    private final boolean allowObfuscation;


    public KeepSpecificationsPanel(JFrame  owner,
                                   boolean markClasses,
                                   boolean markConditionally,
                                   boolean markDescriptorClasses,
                                   boolean allowShrinking,
                                   boolean allowOptimization,
                                   boolean allowObfuscation)
    {
        super(owner, true, true);

        this.markClasses           = markClasses;
        this.markConditionally     = markConditionally;
        this.markDescriptorClasses = markDescriptorClasses;
        this.allowShrinking        = allowShrinking;
        this.allowOptimization     = allowOptimization;
        this.allowObfuscation      = allowObfuscation;
    }


    // Factory methods for ClassSpecificationsPanel.

    protected ClassSpecification createClassSpecification()
    {
        return new KeepClassSpecification(markClasses,
                                          markConditionally,
                                          markDescriptorClasses,
                                          false,
                                          allowShrinking,
                                          allowOptimization,
                                          allowObfuscation,
                                          null,
                                          super.createClassSpecification());
    }


    protected void setClassSpecification(ClassSpecification classSpecification)
    {
        classSpecificationDialog.setKeepSpecification((KeepClassSpecification)classSpecification);
    }


    protected ClassSpecification getClassSpecification()
    {
        return classSpecificationDialog.getKeepSpecification();
    }
}
