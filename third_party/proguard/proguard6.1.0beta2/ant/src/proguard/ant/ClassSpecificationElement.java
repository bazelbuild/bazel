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

import org.apache.tools.ant.BuildException;
import org.apache.tools.ant.types.DataType;
import proguard.*;
import proguard.classfile.*;
import proguard.classfile.util.ClassUtil;

import java.util.*;

/**
 * This DataType represents a class specification in Ant.
 *
 * @author Eric Lafortune
 */
public class ClassSpecificationElement extends DataType
{
    private static final String ANY_CLASS_KEYWORD  = "*";

    private String access;
    private String annotation;
    private String type;
    private String name;
    private String extendsAnnotation;
    private String extends_;
    private List   fieldSpecifications  = new ArrayList();
    private List   methodSpecifications = new ArrayList();


    /**
     * Adds the contents of this class specification element to the given list.
     * @param classSpecifications the class specifications to be extended.
     */
    public void appendTo(List classSpecifications)
    {
        // Get the referenced file set, or else this one.
        ClassSpecificationElement classSpecificationElement = isReference() ?
            (ClassSpecificationElement)getCheckedRef(this.getClass(),
                                                     this.getClass().getName()) :
            this;

        ClassSpecification classSpecification =
            createClassSpecification(classSpecificationElement);

        // Add it to the list.
        classSpecifications.add(classSpecification);
    }


    /**
     * Creates a new class specification corresponding to the contents of this
     * class specification element.
     */
    protected ClassSpecification createClassSpecification(ClassSpecificationElement classSpecificationElement)
    {
        String access            = classSpecificationElement.access;
        String annotation        = classSpecificationElement.annotation;
        String type              = classSpecificationElement.type;
        String name              = classSpecificationElement.name;
        String extendsAnnotation = classSpecificationElement.extendsAnnotation;
        String extends_          = classSpecificationElement.extends_;

        // For backward compatibility, allow a single "*" wildcard to match
        // any class.
        if (name != null &&
            name.equals(ANY_CLASS_KEYWORD))
        {
            name = null;
        }

        ClassSpecification classSpecification =
            new ClassSpecification(null,
                                   requiredAccessFlags(true,  access, type),
                                   requiredAccessFlags(false, access, type),
                                   annotation        != null ? ClassUtil.internalType(annotation)        : null,
                                   name              != null ? ClassUtil.internalClassName(name)         : null,
                                   extendsAnnotation != null ? ClassUtil.internalType(extendsAnnotation) : null,
                                   extends_          != null ? ClassUtil.internalClassName(extends_)     : null);

        for (int index = 0; index < fieldSpecifications.size(); index++)
        {
            classSpecification.addField((MemberSpecification)fieldSpecifications.get(index));
        }

        for (int index = 0; index < methodSpecifications.size(); index++)
        {
            classSpecification.addMethod((MemberSpecification)methodSpecifications.get(index));
        }

        return classSpecification;
    }


    // Ant task attributes.

    public void setAccess(String access)
    {
        this.access = access;
    }


    public void setAnnotation(String annotation)
    {
        this.annotation = annotation;
    }


    public void setType(String type)
    {
        this.type = type;
    }


    public void setName(String name)
    {
        this.name = name;
    }


    public void setExtendsannotation(String extendsAnnotation)
    {
        this.extendsAnnotation = extendsAnnotation;
    }


    public void setExtends(String extends_)
    {
        this.extends_ = extends_;
    }


    public void setImplements(String implements_)
    {
        this.extends_ = implements_;
    }


    // Ant task nested elements.

    public void addConfiguredField(MemberSpecificationElement memberSpecificationElement)
    {
        if (fieldSpecifications == null)
        {
            fieldSpecifications = new ArrayList();
        }

        memberSpecificationElement.appendTo(fieldSpecifications,
                                            false,
                                            false);
    }


    public void addConfiguredMethod(MemberSpecificationElement memberSpecificationElement)
    {
        if (methodSpecifications == null)
        {
            methodSpecifications = new ArrayList();
        }

        memberSpecificationElement.appendTo(methodSpecifications,
                                            true,
                                            false);
    }


    public void addConfiguredConstructor(MemberSpecificationElement memberSpecificationElement)
    {
        if (methodSpecifications == null)
        {
            methodSpecifications = new ArrayList();
        }

        memberSpecificationElement.appendTo(methodSpecifications,
                                            true,
                                            true);
    }


    // Small utility methods.

    private int requiredAccessFlags(boolean set,
                                    String  access,
                                    String  type)
    throws BuildException
    {
        int accessFlags = 0;

        if (access != null)
        {
            StringTokenizer tokenizer = new StringTokenizer(access, " ,");
            while (tokenizer.hasMoreTokens())
            {
                String token = tokenizer.nextToken();

                if (token.startsWith("!") ^ set)
                {
                    String strippedToken = token.startsWith("!") ?
                        token.substring(1) :
                        token;

                    int accessFlag =
                        strippedToken.equals(JavaConstants.ACC_PUBLIC)     ? ClassConstants.ACC_PUBLIC     :
                        strippedToken.equals(JavaConstants.ACC_FINAL)      ? ClassConstants.ACC_FINAL      :
                        strippedToken.equals(JavaConstants.ACC_ABSTRACT)   ? ClassConstants.ACC_ABSTRACT   :
                        strippedToken.equals(JavaConstants.ACC_SYNTHETIC)  ? ClassConstants.ACC_SYNTHETIC  :
                        strippedToken.equals(JavaConstants.ACC_ANNOTATION) ? ClassConstants.ACC_ANNOTATION :
                                                                             0;

                    if (accessFlag == 0)
                    {
                        throw new BuildException("Incorrect class access modifier ["+strippedToken+"]");
                    }

                    accessFlags |= accessFlag;
                }
            }
        }

        if (type != null && (type.startsWith("!") ^ set))
        {
            int accessFlag =
                type.equals("class")                           ? 0                            :
                type.equals(      JavaConstants.ACC_INTERFACE) ||
                type.equals("!" + JavaConstants.ACC_INTERFACE) ? ClassConstants.ACC_INTERFACE :
                type.equals(      JavaConstants.ACC_ENUM)      ||
                type.equals("!" + JavaConstants.ACC_ENUM)      ? ClassConstants.ACC_ENUM      :
                                                                 -1;

            if (accessFlag == -1)
            {
                throw new BuildException("Incorrect class type ["+type+"]");
            }

            accessFlags |= accessFlag;
        }

        return accessFlags;
    }
}
