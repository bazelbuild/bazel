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
package proguard.ant;

import proguard.KeepClassSpecification;

import java.util.List;

/**
 * This DataType represents a class specification in Ant.
 *
 * @author Eric Lafortune
 */
public class KeepSpecificationElement extends ClassSpecificationElement
{
    private boolean markDescriptorClasses;
    private boolean markCodeAttributes;
    private boolean allowShrinking;
    private boolean allowOptimization;
    private boolean allowObfuscation;


    /**
     * Adds the contents of this class specification element to the given list.
     * @param keepSpecifications the class specifications to be extended.
     * @param markClasses        specifies whether to mark the classes.
     * @param markConditionally  specifies whether to mark the classes
     *                           and class members conditionally.
     */
    public void appendTo(List    keepSpecifications,
                         boolean markClasses,
                         boolean markConditionally)
    {
        // Get the referenced file set, or else this one.
        KeepSpecificationElement keepSpecificationElement = isReference() ?
            (KeepSpecificationElement)getCheckedRef(this.getClass(),
                                                    this.getClass().getName()) :
            this;

        KeepClassSpecification keepClassSpecification =
            new KeepClassSpecification(markClasses,
                                       markConditionally,
                                       markDescriptorClasses,
                                       markCodeAttributes,
                                       allowShrinking,
                                       allowOptimization,
                                       allowObfuscation,
                                       null,
                                       createClassSpecification(keepSpecificationElement));

        // Add it to the list.
        keepSpecifications.add(keepClassSpecification);
    }


    // Ant task attributes.

    public void setIncludedescriptorclasses(boolean markDescriptorClasses)
    {
        this.markDescriptorClasses = markDescriptorClasses;
    }


    public void setIncludecode(boolean markCodeAttributes)
    {
        this.markCodeAttributes = markCodeAttributes;
    }


    public void setAllowshrinking(boolean allowShrinking)
    {
        this.allowShrinking = allowShrinking;
    }


    public void setAllowoptimization(boolean allowOptimization)
    {
        this.allowOptimization = allowOptimization;
    }


    public void setAllowobfuscation(boolean allowObfuscation)
    {
        this.allowObfuscation = allowObfuscation;
    }
}
