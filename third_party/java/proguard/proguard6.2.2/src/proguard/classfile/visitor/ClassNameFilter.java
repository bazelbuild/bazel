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
package proguard.classfile.visitor;

import proguard.classfile.*;
import proguard.util.*;

import java.util.List;

/**
 * This <code>ClassVisitor</code> delegates its visits to another given
 * <code>ClassVisitor</code>, but only when the visited class has a name that
 * matches a given regular expression.
 *
 * @author Eric Lafortune
 */
public class ClassNameFilter implements ClassVisitor
{
    private final StringMatcher regularExpressionMatcher;
    private final ClassVisitor  classVisitor;


    /**
     * Creates a new ClassNameFilter.
     * @param regularExpression      the regular expression against which class
     *                               names will be matched.
     * @param classVisitor           the <code>ClassVisitor</code> to which
     *                               visits will be delegated.
     */
    public ClassNameFilter(String       regularExpression,
                           ClassVisitor classVisitor)
    {
        this(regularExpression, null, classVisitor);
    }


    /**
     * Creates a new ClassNameFilter.
     * @param regularExpression      the regular expression against which class
     *                               names will be matched.
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     *                               the wildcards.
     * @param classVisitor           the <code>ClassVisitor</code> to which
     *                               visits will be delegated.
     */
    public ClassNameFilter(String       regularExpression,
                           List         variableStringMatchers,
                           ClassVisitor classVisitor)
    {
        this(new ListParser(new ClassNameParser(variableStringMatchers)).parse(regularExpression),
             classVisitor);
    }


    /**
     * Creates a new ClassNameFilter.
     * @param regularExpression      the regular expression against which class
     *                               names will be matched.
     * @param classVisitor           the <code>ClassVisitor</code> to which
     *                               visits will be delegated.
     */
    public ClassNameFilter(List         regularExpression,
                           ClassVisitor classVisitor)
    {
        this(regularExpression, null, classVisitor);
    }


    /**
     * Creates a new ClassNameFilter.
     * @param regularExpression      the regular expression against which class
     *                               names will be matched.
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     *                               the wildcards.
     * @param classVisitor           the <code>ClassVisitor</code> to which
     *                               visits will be delegated.
     */
    public ClassNameFilter(List         regularExpression,
                           List         variableStringMatchers,
                           ClassVisitor classVisitor)
    {
        this(new ListParser(new ClassNameParser(variableStringMatchers)).parse(regularExpression),
             classVisitor);
    }


    /**
     * Creates a new ClassNameFilter.
     * @param regularExpressionMatcher the string matcher against which
     *                                 class names will be matched.
     * @param classVisitor             the <code>ClassVisitor</code> to which
     *                                 visits will be delegated.
     */
    public ClassNameFilter(StringMatcher regularExpressionMatcher,
                           ClassVisitor  classVisitor)
    {
        this.regularExpressionMatcher = regularExpressionMatcher;
        this.classVisitor             = classVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (accepted(programClass.getName()))
        {
            classVisitor.visitProgramClass(programClass);
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        if (accepted(libraryClass.getName()))
        {
            classVisitor.visitLibraryClass(libraryClass);
        }
    }


    // Small utility methods.

    private boolean accepted(String name)
    {
        return regularExpressionMatcher.matches(name);
    }
}
