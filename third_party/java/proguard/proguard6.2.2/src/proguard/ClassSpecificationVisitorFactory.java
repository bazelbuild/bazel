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
package proguard;

import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.visitor.*;
import proguard.util.*;

import java.util.List;

/**
 * This factory creates visitors to efficiently travel to specified classes and
 * class members.
 *
 * @author Eric Lafortune
 */
public class ClassSpecificationVisitorFactory
{
    /**
     * Constructs a ClassPoolVisitor to efficiently travel to the specified
     * classes, class members, and attributes.
     *
     * @param classSpecifications the list of ClassSpecification instances
     *                            that specify the classes and class members
     *                            to visit.
     * @param classVisitor        an optional ClassVisitor to be applied to
     *                            all classes.
     * @param memberVisitor       an optional MemberVisitor to be applied to
     *                            matching fields and methods.
     */
    public ClassPoolVisitor createClassPoolVisitor(List          classSpecifications,
                                                   ClassVisitor  classVisitor,
                                                   MemberVisitor memberVisitor)
    {
        return createClassPoolVisitor(classSpecifications,
                                      classVisitor,
                                      memberVisitor,
                                      memberVisitor,
                                      null);
    }


    /**
     * Constructs a ClassPoolVisitor to efficiently travel to the specified
     * classes and class members.
     *
     * @param classSpecifications the list of ClassSpecification instances
     *                            that specify the classes and class members
     *                            to visit.
     * @param classVisitor        an optional ClassVisitor to be applied to
     *                            all classes.
     * @param fieldVisitor        an optional MemberVisitor to be applied to
     *                            matching fields.
     * @param methodVisitor       an optional MemberVisitor to be applied to
     *                            matching methods.
     * @param attributeVisitor    an optional AttributeVisitor to be applied
     *                            to matching attributes.
     */
    public ClassPoolVisitor createClassPoolVisitor(List             classSpecifications,
                                                   ClassVisitor     classVisitor,
                                                   MemberVisitor    fieldVisitor,
                                                   MemberVisitor    methodVisitor,
                                                   AttributeVisitor attributeVisitor)
    {
        MultiClassPoolVisitor multiClassPoolVisitor = new MultiClassPoolVisitor();

        if (classSpecifications != null)
        {
            for (int index = 0; index < classSpecifications.size(); index++)
            {
                ClassSpecification classSpecification =
                    (ClassSpecification)classSpecifications.get(index);

                multiClassPoolVisitor.addClassPoolVisitor(
                    createClassPoolVisitor(classSpecification,
                                           classVisitor,
                                           fieldVisitor,
                                           methodVisitor,
                                           attributeVisitor,
                                           null));
            }
        }

        return multiClassPoolVisitor;
    }


    /**
     * Constructs a ClassPoolVisitor to efficiently travel to the specified
     * classes, class members, and attributes.
     *
     * @param classSpecification     the specifications of the class(es) and class
     *                               members to visit.
     * @param classVisitor           an optional ClassVisitor to be applied to
     *                               matching classes.
     * @param fieldVisitor           an optional MemberVisitor to be applied to
     *                               matching fields.
     * @param methodVisitor          an optional MemberVisitor to be applied to
     *                               matching methods.
     * @param attributeVisitor       an optional AttributeVisitor to be applied
     *                               to matching attributes.
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     *                               the wildcards.
     */
    protected ClassPoolVisitor createClassPoolVisitor(ClassSpecification classSpecification,
                                                      ClassVisitor       classVisitor,
                                                      MemberVisitor      fieldVisitor,
                                                      MemberVisitor      methodVisitor,
                                                      AttributeVisitor   attributeVisitor,
                                                      List               variableStringMatchers)
    {
        String className             = classSpecification.className;
        String annotationType        = classSpecification.annotationType;
        String extendsAnnotationType = classSpecification.extendsAnnotationType;
        String extendsClassName      = classSpecification.extendsClassName;

        // We explicitly need to match a wildcard class name, so it can be
        // referenced through its variable string matcher.
        if (className == null)
        {
            className = "**";
        }

        // We need to parse the class names before any class member names, to
        // make sure the list of variable string matchers is filled out in the
        // right order.
        StringMatcher annotationTypeMatcher = annotationType == null ? null :
            new ListParser(new ClassNameParser(variableStringMatchers)).parse(annotationType);

        StringMatcher classNameMatcher =
            new ListParser(new ClassNameParser(variableStringMatchers)).parse(className);

        StringMatcher extendsAnnotationTypeMatcher = extendsAnnotationType == null ? null :
            new ListParser(new ClassNameParser(variableStringMatchers)).parse(extendsAnnotationType);

        StringMatcher extendsClassNameMatcher = extendsClassName == null ? null :
            new ListParser(new ClassNameParser(variableStringMatchers)).parse(extendsClassName);

        // Combine both visitors.
        ClassVisitor combinedClassVisitor =
            createCombinedClassVisitor(classSpecification.attributeNames,
                                       classSpecification.fieldSpecifications,
                                       classSpecification.methodSpecifications,
                                       classVisitor,
                                       fieldVisitor,
                                       methodVisitor,
                                       attributeVisitor,
                                       variableStringMatchers);

        // If the class name has wildcards, only visit classes with matching names.
        if (extendsAnnotationType != null ||
            extendsClassName      != null ||
            containsWildCards(className))
        {
            combinedClassVisitor =
                new ClassNameFilter(classNameMatcher, combinedClassVisitor);

            // We'll have to visit all classes now.
            className = null;
        }

        // If specified, only visit classes with the right annotation.
        if (annotationType != null)
        {
            combinedClassVisitor =
                new AllAttributeVisitor(
                new AllAnnotationVisitor(
                new AnnotationTypeFilter(annotationTypeMatcher,
                new AnnotationToAnnotatedClassVisitor(combinedClassVisitor))));
        }

        // If specified, only visit classes with the right access flags.
        if (classSpecification.requiredSetAccessFlags   != 0 ||
            classSpecification.requiredUnsetAccessFlags != 0)
        {
            combinedClassVisitor =
                new ClassAccessFilter(classSpecification.requiredSetAccessFlags,
                                      classSpecification.requiredUnsetAccessFlags,
                                      combinedClassVisitor);
        }

        // If it's specified, start visiting from the extended class.
        if (extendsAnnotationType != null ||
            extendsClassName      != null)
        {
            // Start visiting from the extended class.
            combinedClassVisitor =
                new ClassHierarchyTraveler(false, false, false, true,
                                           combinedClassVisitor);

            // If specified, only visit extended classes with the right annotation.
            if (extendsAnnotationType != null)
            {
                combinedClassVisitor =
                    new AllAttributeVisitor(
                    new AllAnnotationVisitor(
                    new AnnotationTypeFilter(extendsAnnotationTypeMatcher,
                    new AnnotationToAnnotatedClassVisitor(combinedClassVisitor))));
            }

            // If specified, only visit extended classes with matching names.
            if (extendsClassName != null)
            {
                // If wildcarded, only visit extended classes with matching names.
                if (containsWildCards(extendsClassName))
                {
                    combinedClassVisitor =
                        new ClassNameFilter(extendsClassNameMatcher,
                                            combinedClassVisitor);
                }
                else
                {
                    // Start visiting from the named extended class.
                    className = extendsClassName;
                }
            }
        }

        // If specified, visit a single named class, otherwise visit all classes.
        return className != null ?
            new NamedClassVisitor(combinedClassVisitor, className) :
            new AllClassVisitor(combinedClassVisitor);
    }


    /**
     * Constructs a ClassVisitor to efficiently delegate to the given ClassVisitor
     * and travel to the specified class members and attributes.
     * @param attributeNames         optional names (with wildcards) of class
     *                               attributes to visit.
     * @param fieldSpecifications    optional specifications of the fields to
     *                               visit.
     * @param methodSpecifications   optional specifications of the methods to
     *                               visit.
     * @param classVisitor           an optional ClassVisitor to be applied to
     *                               all classes.
     * @param fieldVisitor           an optional MemberVisitor to be applied to
     *                               matching fields.
     * @param methodVisitor          an optional MemberVisitor to be applied to
     *                               matching methods.
     * @param attributeVisitor       an optional AttributeVisitor to be applied
     *                               to matching attributes.
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     */
    protected ClassVisitor createCombinedClassVisitor(List             attributeNames,
                                                      List             fieldSpecifications,
                                                      List             methodSpecifications,
                                                      ClassVisitor     classVisitor,
                                                      MemberVisitor    fieldVisitor,
                                                      MemberVisitor    methodVisitor,
                                                      AttributeVisitor attributeVisitor,
                                                      List             variableStringMatchers)
    {
        // Don't visit any members if there aren't any member specifications.
        if (fieldSpecifications  == null)
        {
            fieldVisitor  = null;
        }

        if (methodSpecifications == null)
        {
            methodVisitor = null;
        }

        // The class visitor for classes and their members.
        MultiClassVisitor multiClassVisitor = new MultiClassVisitor();

        // If specified, let the class visitor visit the class itself.
        if (classVisitor != null)
        {
            // This class visitor may be the only one.
            if (fieldVisitor     == null &&
                methodVisitor    == null &&
                attributeVisitor == null)
            {
                return classVisitor;
            }

            multiClassVisitor.addClassVisitor(classVisitor);
        }

        // If specified, let the attribute visitor visit the class attributes.
        if (attributeVisitor != null)
        {
            // If specified, only visit attributes with the right names.
            if (attributeNames != null)
            {
                attributeVisitor =
                    new AttributeNameFilter(attributeNames, attributeVisitor);
            }

            multiClassVisitor.addClassVisitor(new AllAttributeVisitor(attributeVisitor));
        }

        // If specified, let the member info visitor visit the class members.
        if (fieldVisitor  != null ||
            methodVisitor != null)
        {
            ClassVisitor memberClassVisitor =
                createClassVisitor(fieldSpecifications,
                                   methodSpecifications,
                                   fieldVisitor,
                                   methodVisitor,
                                   attributeVisitor,
                                   variableStringMatchers);

            // This class visitor may be the only one.
            if (classVisitor == null)
            {
                return memberClassVisitor;
            }

            multiClassVisitor.addClassVisitor(memberClassVisitor);
        }

        return multiClassVisitor;
    }


    /**
     * Constructs a ClassVisitor to efficiently travel to the specified class
     * members.
     *
     * @param fieldSpecifications    the specifications of the fields to visit.
     * @param methodSpecifications   the specifications of the methods to visit.
     * @param fieldVisitor           an optional MemberVisitor to be applied to
     *                               matching fields.
     * @param methodVisitor          an optional MemberVisitor to be applied to
     *                               matching methods.
     * @param attributeVisitor       an optional AttributeVisitor to be applied
     *                               to matching attributes.
     * @param variableStringMatchers an optional mutable list of
     *                               VariableStringMatcher instances that match
     *                               the wildcards.
     */
    private ClassVisitor createClassVisitor(List             fieldSpecifications,
                                            List             methodSpecifications,
                                            MemberVisitor    fieldVisitor,
                                            MemberVisitor    methodVisitor,
                                            AttributeVisitor attributeVisitor,
                                            List             variableStringMatchers)
    {
        MultiClassVisitor multiClassVisitor = new MultiClassVisitor();

        addMemberVisitors(fieldSpecifications,  true,  multiClassVisitor, fieldVisitor,  attributeVisitor, variableStringMatchers);
        addMemberVisitors(methodSpecifications, false, multiClassVisitor, methodVisitor, attributeVisitor, variableStringMatchers);

        // Mark the class member in this class and in super classes.
        return new ClassHierarchyTraveler(true, true, false, false,
                                          multiClassVisitor);
    }


    /**
     * Adds elements to the given MultiClassVisitor, to apply the given
     * MemberVisitor to all class members that match the given List
     * of options (of the given type).
     */
    private void addMemberVisitors(List              memberSpecifications,
                                   boolean           isField,
                                   MultiClassVisitor multiClassVisitor,
                                   MemberVisitor     memberVisitor,
                                   AttributeVisitor  attributeVisitor,
                                   List              variableStringMatchers)
    {
        if (memberSpecifications != null)
        {
            for (int index = 0; index < memberSpecifications.size(); index++)
            {
                MemberSpecification memberSpecification =
                    (MemberSpecification)memberSpecifications.get(index);

                multiClassVisitor.addClassVisitor(
                    createNonTestingClassVisitor(memberSpecification,
                                                 isField,
                                                 memberVisitor,
                                                 attributeVisitor,
                                                 variableStringMatchers));
            }
        }
    }


    /**
     * Creates a new ClassVisitor to efficiently travel to the specified class
     * members and attributes.
     *
     * @param memberSpecification    the specification of the class member(s) to
     *                               visit.
     * @param memberVisitor          the MemberVisitor to be applied to matching
     *                               class member(s).
     * @param variableStringMatchers a mutable list of VariableStringMatcher
     *                               instances that match the wildcards.
     */
    protected ClassVisitor createNonTestingClassVisitor(MemberSpecification memberSpecification,
                                                        boolean             isField,
                                                        MemberVisitor       memberVisitor,
                                                        AttributeVisitor    attributeVisitor,
                                                        List                variableStringMatchers)
    {
        return createClassVisitor(memberSpecification,
                                  isField,
                                  memberVisitor,
                                  attributeVisitor,
                                  variableStringMatchers);
    }


    /**
     * Constructs a ClassPoolVisitor that conditionally applies the given
     * ClassPoolVisitor for all classes that match the given class
     * specification.
     */
    protected ClassPoolVisitor createClassTester(ClassSpecification classSpecification,
                                                 ClassPoolVisitor   classPoolVisitor,
                                                 List               variableStringMatchers)
    {
        ClassPoolClassVisitor classPoolClassVisitor =
            new ClassPoolClassVisitor(classPoolVisitor);

        // Parse the class condition.
        ClassPoolVisitor conditionalClassTester =
            createClassTester(classSpecification,
                              (ClassVisitor)classPoolClassVisitor,
                              variableStringMatchers);

        // The ClassPoolClassVisitor first needs to visit the class pool
        // and then its classes.
        return new MultiClassPoolVisitor(
               new ClassPoolVisitor[]
               {
                   classPoolClassVisitor,
                   conditionalClassTester
               });
    }


    /**
     * Constructs a ClassPoolVisitor that conditionally applies the given
     * ClassVisitor to all classes that match the given class specification.
     */
    protected ClassPoolVisitor createClassTester(ClassSpecification classSpecification,
                                                 ClassVisitor       classVisitor,
                                                 List               variableStringMatchers)
    {
        // Create a placeholder for the class visitor that tests class
        // members.
        MultiClassVisitor conditionalMemberTester =
            new MultiClassVisitor();

        // Parse the class condition.
        ClassPoolVisitor conditionalClassTester =
            createClassPoolVisitor(classSpecification,
                                   conditionalMemberTester,
                                   null,
                                   null,
                                   null,
                                   variableStringMatchers);

        // Parse the member conditions and add the result to the placeholder.
        conditionalMemberTester.addClassVisitor(
            createClassMemberTester(classSpecification.fieldSpecifications,
                                    classSpecification.methodSpecifications,
                                    classVisitor,
                                    variableStringMatchers));

        return conditionalClassTester;
    }


    /**
     * Constructs a ClassVisitor that conditionally applies the given
     * ClassVisitor to all classes that contain the given class members.
     */
    private ClassVisitor createClassMemberTester(List         fieldSpecifications,
                                                 List         methodSpecifications,
                                                 ClassVisitor classVisitor,
                                                 List         variableStringMatchers)
    {
        // Create a linked list of conditional visitors, for fields and for
        // methods.
        return createClassMemberTester(fieldSpecifications,
                                       true,
               createClassMemberTester(methodSpecifications,
                                       false,
                                       classVisitor, variableStringMatchers),
                                       variableStringMatchers);
    }


    /**
     * Constructs a ClassVisitor that conditionally applies the given
     * ClassVisitor to all classes that contain the given List of class
     * members (of the given type).
     */
    private ClassVisitor createClassMemberTester(List         memberSpecifications,
                                                 boolean      isField,
                                                 ClassVisitor classVisitor,
                                                 List         variableStringMatchers)
    {
        // Create a linked list of conditional visitors.
        if (memberSpecifications != null)
        {
            for (int index = 0; index < memberSpecifications.size(); index++)
            {
                MemberSpecification memberSpecification =
                    (MemberSpecification)memberSpecifications.get(index);

                classVisitor =
                    createClassVisitor(memberSpecification,
                                       isField,
                                       new MemberToClassVisitor(classVisitor),
                                       null,
                                       variableStringMatchers);
            }
        }

        return classVisitor;
    }


    /**
     * Creates a new ClassVisitor to efficiently travel to the specified class
     * members and attributes.
     *
     * @param memberSpecification    the specification of the class member(s) to
     *                               visit.
     * @param memberVisitor          the MemberVisitor to be applied to matching
     *                               class member(s).
     * @param variableStringMatchers a mutable list of VariableStringMatcher
     *                               instances that match the wildcards.
     */
    private ClassVisitor createClassVisitor(MemberSpecification memberSpecification,
                                            boolean             isField,
                                            MemberVisitor       memberVisitor,
                                            AttributeVisitor    attributeVisitor,
                                            List                variableStringMatchers)
    {
        String annotationType = memberSpecification.annotationType;
        String name           = memberSpecification.name;
        String descriptor     = memberSpecification.descriptor;
        List   attributeNames = memberSpecification.attributeNames;

        // We need to parse the names before the descriptors, to make sure the
        // list of variable string matchers is filled out in the right order.
        StringMatcher annotationTypeMatcher = annotationType == null ? null :
            new ListParser(new ClassNameParser(variableStringMatchers)).parse(annotationType);

        StringMatcher nameMatcher = name == null ? null :
            new ListParser(new NameParser(variableStringMatchers)).parse(name);

        StringMatcher descriptorMatcher = descriptor == null ? null :
            new ListParser(new ClassNameParser(variableStringMatchers)).parse(descriptor);

        StringMatcher attributesMatcher = attributeNames == null ? null :
            new ListParser(new NameParser(variableStringMatchers)).parse(attributeNames);

        // If specified, let the attribute visitor visit the class member
        // attributes.
        if (attributeVisitor != null)
        {
            // If specified, only visit attributes with the right names.
            if (attributesMatcher != null)
            {
                attributeVisitor =
                    new AttributeNameFilter(attributesMatcher, attributeVisitor);
            }

            memberVisitor =
                new MultiMemberVisitor(
                new MemberVisitor[]
                {
                    memberVisitor,
                    new AllAttributeVisitor(attributeVisitor)
                });
        }

        // If specified, only visit class members with the right annotation.
        if (memberSpecification.annotationType != null)
        {
            memberVisitor =
                new AllAttributeVisitor(
                new AllAnnotationVisitor(
                new AnnotationTypeFilter(annotationTypeMatcher,
                new AnnotationToAnnotatedMemberVisitor(memberVisitor))));
        }

        // If any access flags are specified, only visit matching class members.
        if (memberSpecification.requiredSetAccessFlags   != 0 ||
            memberSpecification.requiredUnsetAccessFlags != 0)
        {
            memberVisitor =
                new MemberAccessFilter(memberSpecification.requiredSetAccessFlags,
                                       memberSpecification.requiredUnsetAccessFlags,
                                       memberVisitor);
        }

        // Are the name and descriptor fully specified?
        if (name       != null       &&
            descriptor != null       &&
            !containsWildCards(name) &&
            !containsWildCards(descriptor))
        {
            // Somewhat more efficiently, visit a single named class member.
            return isField ?
                new NamedFieldVisitor(name, descriptor, memberVisitor) :
                new NamedMethodVisitor(name, descriptor, memberVisitor);
        }

        // If specified, only visit class members with the right descriptors.
        if (descriptorMatcher != null)
        {
            memberVisitor =
                new MemberDescriptorFilter(descriptorMatcher, memberVisitor);
        }

        // If specified, only visit class members with the right names.
        if (name != null)
        {
            memberVisitor =
                new MemberNameFilter(nameMatcher, memberVisitor);
        }

        // Visit all class members, filtering the matching ones.
        return isField ?
            new AllFieldVisitor(memberVisitor) :
            new AllMethodVisitor(memberVisitor);
    }


    // Small utility methods.

    /**
     * Returns whether the given string contains a wild card.
     */
    private boolean containsWildCards(String string)
    {
        return string != null &&
            (string.indexOf('!')   >= 0 ||
             string.indexOf('*')   >= 0 ||
             string.indexOf('?')   >= 0 ||
             string.indexOf('%')   >= 0 ||
             string.indexOf(',')   >= 0 ||
             string.indexOf("///") >= 0 ||
             containsWildCardReferences(string));
    }


    /**
     * Returns whether the given string contains a numeric reference to a
     * wild card ("<n>").
     */
    private boolean containsWildCardReferences(String string)
    {
        int openIndex = string.indexOf('<');
        if (openIndex < 0)
        {
            return false;
        }

        int closeIndex = string.indexOf('>', openIndex + 1);
        if (closeIndex < 0)
        {
            return false;
        }

        try
        {
            Integer.parseInt(string.substring(openIndex + 1, closeIndex));
        }
        catch (NumberFormatException e)
        {
            return false;
        }

        return true;
    }
}
