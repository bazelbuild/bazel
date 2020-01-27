/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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

/**
 * This class represents a keep option with class specification.
 *
 * @author Eric Lafortune
 */
public class KeepClassSpecification extends ClassSpecification
{
    public final boolean markClasses;
    public final boolean markConditionally;
    public final boolean markDescriptorClasses;
    public final boolean allowShrinking;
    public final boolean allowOptimization;
    public final boolean allowObfuscation;


    /**
     * Creates a new KeepClassSpecification for all possible classes.
     * @param markClasses           specifies whether to mark the classes.
     *                              If false, only class members are marked.
     *                              If true, the classes are marked as well.
     * @param markConditionally     specifies whether to mark the classes and
     *                              class members conditionally. If true,
     *                              classes and class members are marked, on
     *                              the condition that all specified class
     *                              members are present.
     * @param markDescriptorClasses specifies whether to mark the classes in
     *                              the descriptors of the marked class members.
     * @param allowShrinking        specifies whether shrinking is allowed.
     * @param allowOptimization     specifies whether optimization is allowed.
     * @param allowObfuscation      specifies whether obfuscation is allowed.
     */
    public KeepClassSpecification(boolean markClasses,
                                  boolean markConditionally,
                                  boolean markDescriptorClasses,
                                  boolean allowShrinking,
                                  boolean allowOptimization,
                                  boolean allowObfuscation)
    {
        this.markClasses           = markClasses;
        this.markConditionally     = markConditionally;
        this.markDescriptorClasses = markDescriptorClasses;
        this.allowShrinking        = allowShrinking;
        this.allowOptimization     = allowOptimization;
        this.allowObfuscation      = allowObfuscation;
    }


    /**
     * Creates a new KeepClassSpecification.
     * @param markClasses           specifies whether to mark the classes.
     *                              If false, only class members are marked.
     *                              If true, the classes are marked as well.
     * @param markConditionally     specifies whether to mark the classes and
     *                              class members conditionally. If true,
     *                              classes and class members are marked, on
     *                              the condition that all specified class
     *                              members are present.
     * @param markDescriptorClasses specifies whether to mark the classes in
     *                              the descriptors of the marked class members.
     * @param allowShrinking        specifies whether shrinking is allowed.
     * @param allowOptimization     specifies whether optimization is allowed.
     * @param allowObfuscation      specifies whether obfuscation is allowed.
     * @param classSpecification    the specification of classes and class
     *                              members.
     */
    public KeepClassSpecification(boolean            markClasses,
                                  boolean            markConditionally,
                                  boolean            markDescriptorClasses,
                                  boolean            allowShrinking,
                                  boolean            allowOptimization,
                                  boolean            allowObfuscation,
                                  ClassSpecification classSpecification)
    {
        super(classSpecification);

        this.markClasses           = markClasses;
        this.markConditionally     = markConditionally;
        this.markDescriptorClasses = markDescriptorClasses;
        this.allowShrinking        = allowShrinking;
        this.allowOptimization     = allowOptimization;
        this.allowObfuscation      = allowObfuscation;
    }


    // Implementations for Object.

    public boolean equals(Object object)
    {
        if (object == null ||
            this.getClass() != object.getClass())
        {
            return false;
        }

        KeepClassSpecification other = (KeepClassSpecification)object;
        return
            this.markClasses           == other.markClasses           &&
            this.markConditionally     == other.markConditionally     &&
            this.markDescriptorClasses == other.markDescriptorClasses &&
            this.allowShrinking        == other.allowShrinking        &&
            this.allowOptimization     == other.allowOptimization     &&
            this.allowObfuscation      == other.allowObfuscation      &&
            super.equals(other);
    }

    public int hashCode()
    {
        return
            (markClasses           ? 0 :  1) ^
            (markConditionally     ? 0 :  2) ^
            (markDescriptorClasses ? 0 :  4) ^
            (allowShrinking        ? 0 :  8) ^
            (allowOptimization     ? 0 : 16) ^
            (allowObfuscation      ? 0 : 32) ^
            super.hashCode();
    }

    public Object clone()
    {
//        try
//        {
            return super.clone();
//        }
//        catch (CloneNotSupportedException e)
//        {
//            return null;
//        }
    }
}
