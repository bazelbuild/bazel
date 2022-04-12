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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.editor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.optimize.KeepMarker;
import proguard.optimize.info.*;
import proguard.util.*;

import java.util.*;

/**
 * This ClassVisitor inlines the classes that it visits in a given target class,
 * whenever possible.
 *
 * @see RetargetedInnerClassAttributeRemover
 * @see TargetClassChanger
 * @see ClassReferenceFixer
 * @see MemberReferenceFixer
 * @see AccessFixer
 * @author Eric Lafortune
 */
public class ClassMerger
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor
{
    //*
    private static final boolean DEBUG   = false;
    private static final boolean DETAILS = false;
    /*/
    private static       boolean DEBUG   = System.getProperty("cm") != null;
    private static       boolean DETAILS = System.getProperty("cmd") != null;
    //*/


    private final ProgramClass targetClass;
    private final boolean      allowAccessModification;
    private final boolean      mergeInterfacesAggressively;
    private final boolean      mergeWrapperClasses;
    private final ClassVisitor extraClassVisitor;

    private final MemberVisitor fieldOptimizationInfoCopier = new FieldOptimizationInfoCopier();


    /**
     * Creates a new ClassMerger that will merge classes into the given target
     * class.
     * @param targetClass                 the class into which all visited
     *                                    classes will be merged.
     * @param allowAccessModification     specifies whether the access modifiers
     *                                    of classes can be changed in order to
     *                                    merge them.
     * @param mergeInterfacesAggressively specifies whether interfaces may
     *                                    be merged aggressively.
     */
    public ClassMerger(ProgramClass targetClass,
                       boolean      allowAccessModification,
                       boolean      mergeInterfacesAggressively,
                       boolean      mergeWrapperClasses)
    {
        this(targetClass,
             allowAccessModification,
             mergeInterfacesAggressively,
             mergeWrapperClasses,
             null);
    }


    /**
     * Creates a new ClassMerger that will merge classes into the given target
     * class.
     * @param targetClass                 the class into which all visited
     *                                    classes will be merged.
     * @param allowAccessModification     specifies whether the access modifiers
     *                                    of classes can be changed in order to
     *                                    merge them.
     * @param mergeInterfacesAggressively specifies whether interfaces may
     *                                    be merged aggressively.
     * @param extraClassVisitor           an optional extra visitor for all
     *                                    merged classes.
     */
    public ClassMerger(ProgramClass targetClass,
                       boolean      allowAccessModification,
                       boolean      mergeInterfacesAggressively,
                       boolean      mergeWrapperClasses,
                       ClassVisitor extraClassVisitor)
    {
        this.targetClass                 = targetClass;
        this.allowAccessModification     = allowAccessModification;
        this.mergeInterfacesAggressively = mergeInterfacesAggressively;
        this.mergeWrapperClasses         = mergeWrapperClasses;
        this.extraClassVisitor           = extraClassVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        //final String CLASS_NAME = "abc/Def";
        //DEBUG = programClass.getName().equals(CLASS_NAME) ||
        //        targetClass.getName().equals(CLASS_NAME);

        // TODO: Remove this when the class merger has stabilized.
        // Catch any unexpected exceptions from the actual visiting method.
        try
        {
            visitProgramClass0(programClass);
        }
        catch (RuntimeException ex)
        {
            System.err.println("Unexpected error while merging classes:");
            System.err.println("  Class        = ["+programClass.getName()+"]");
            System.err.println("  Target class = ["+targetClass.getName()+"]");
            System.err.println("  Exception    = ["+ex.getClass().getName()+"] ("+ex.getMessage()+")");

            if (DEBUG)
            {
                programClass.accept(new ClassPrinter());
                targetClass.accept(new ClassPrinter());
            }

            throw ex;
        }
    }

    public void visitProgramClass0(ProgramClass programClass)
    {
        if (!programClass.equals(targetClass) &&

            // Don't merge classes that must be preserved.
            !KeepMarker.isKept(programClass) &&
            !KeepMarker.isKept(targetClass) &&

            // Only merge classes that haven't been retargeted yet.
            getTargetClass(programClass) == null &&
            getTargetClass(targetClass)  == null &&

            // Don't merge annotation classes, with all their reflection and
            // infinite recursion.
            (programClass.getAccessFlags() & ClassConstants.ACC_ANNOTATION) == 0 &&

            (!DETAILS || print(programClass, "Version?")) &&

            // Only merge classes with equal class versions.
            programClass.u4version == targetClass.u4version &&

            (!DETAILS || print(programClass, "Package visibility?")) &&

            // Only merge classes if we can change the access permissions, or
            // if they are in the same package, or
            // if they are public and don't contain or invoke package visible
            // class members.
            (allowAccessModification                                                        ||
             ((programClass.getAccessFlags() &
               targetClass.getAccessFlags()  &
               ClassConstants.ACC_PUBLIC) != 0 &&
              !PackageVisibleMemberContainingClassMarker.containsPackageVisibleMembers(programClass) &&
              !PackageVisibleMemberInvokingClassMarker.invokesPackageVisibleMembers(programClass)) ||
             ClassUtil.internalPackageName(programClass.getName()).equals(
             ClassUtil.internalPackageName(targetClass.getName()))) &&

            (!DETAILS || print(programClass, "Interface/abstract/single?")) &&

            // Only merge two classes or two interfaces or two abstract classes,
            // or a single implementation into its interface.
            ((programClass.getAccessFlags() &
              (ClassConstants.ACC_INTERFACE |
               ClassConstants.ACC_ABSTRACT)) ==
             (targetClass.getAccessFlags()  &
              (ClassConstants.ACC_INTERFACE |
               ClassConstants.ACC_ABSTRACT)) ||
             (isOnlySubClass(programClass, targetClass) &&
              programClass.getSuperClass() != null      &&
              (programClass.getSuperClass().equals(targetClass) ||
               programClass.getSuperClass().equals(targetClass.getSuperClass())))) &&

            (!DETAILS || print(programClass, "Indirect implementation?")) &&

            // One class must not implement the other class indirectly.
            !indirectlyImplementedInterfaces(programClass).contains(targetClass) &&
            !targetClass.extendsOrImplements(programClass) &&

            (!DETAILS || print(programClass, "Interfaces same subinterfaces?")) &&

            // Interfaces must have exactly the same subinterfaces, not
            // counting themselves, to avoid any loops in the interface
            // hierarchy.
            ((programClass.getAccessFlags() & ClassConstants.ACC_INTERFACE) == 0 ||
             (targetClass.getAccessFlags()  & ClassConstants.ACC_INTERFACE) == 0 ||
             subInterfaces(programClass, targetClass).equals(subInterfaces(targetClass, programClass))) &&

            (!DETAILS || print(programClass, "Same initialized superclasses?")) &&

            // The two classes must have the same superclasses and interfaces
            // with static initializers.
            sideEffectSuperClasses(programClass).equals(sideEffectSuperClasses(targetClass)) &&

            (!DETAILS || print(programClass, "Same instanceofed superclasses?")) &&

            // The two classes must have the same superclasses and interfaces
            // that are tested with 'instanceof'.
            instanceofedSuperClasses(programClass).equals(instanceofedSuperClasses(targetClass)) &&

            (!DETAILS || print(programClass, "Same caught superclasses?")) &&

            // The two classes must have the same superclasses that are caught
            // as exceptions.
            caughtSuperClasses(programClass).equals(caughtSuperClasses(targetClass)) &&

            (!DETAILS || print(programClass, "Not .classed?")) &&

            // The two classes must not both be part of a .class construct.
            !(DotClassMarker.isDotClassed(programClass) &&
              DotClassMarker.isDotClassed(targetClass)) &&

            (!DETAILS || print(programClass, "No clashing fields?")) &&

            // The classes must not have clashing fields.
            (mergeWrapperClasses ||
             !haveAnyIdenticalFields(programClass, targetClass)) &&

            (!DETAILS || print(programClass, "No unwanted fields?")) &&

            // The two classes must not introduce any unwanted fields.
            (mergeWrapperClasses ||
             !introducesUnwantedFields(programClass, targetClass) &&
             !introducesUnwantedFields(targetClass, programClass)) &&

            (!DETAILS || print(programClass, "No shadowed fields?")) &&

            // The two classes must not shadow each others fields.
            (mergeWrapperClasses ||
             !shadowsAnyFields(programClass, targetClass) &&
             !shadowsAnyFields(targetClass, programClass)) &&

            (!DETAILS || print(programClass, "No clashing methods?")) &&

            // The classes must not have clashing methods.
            !haveAnyIdenticalMethods(programClass, targetClass) &&

            (!DETAILS || print(programClass, "No abstract methods?")) &&

            // The classes must not introduce abstract methods, unless
            // explicitly allowed.
            (mergeInterfacesAggressively ||
             (!introducesUnwantedAbstractMethods(programClass, targetClass) &&
              !introducesUnwantedAbstractMethods(targetClass, programClass))) &&

            (!DETAILS || print(programClass, "No overridden methods?")) &&

            // The classes must not override each others concrete methods.
            !overridesAnyMethods(programClass, targetClass) &&
            !overridesAnyMethods(targetClass, programClass) &&

            (!DETAILS || print(programClass, "No shadowed methods?")) &&

            // The classes must not shadow each others non-private methods.
            !shadowsAnyMethods(programClass, targetClass) &&
            !shadowsAnyMethods(targetClass, programClass) &&

            (!DETAILS || print(programClass, "No type variables/parameterized types?")) &&

            // The two classes must not have a signature attribute as type variables
            // and/or parameterized types can not always be merged.
            !hasSignatureAttribute(programClass) &&
            !hasSignatureAttribute(targetClass)  &&

            (!DETAILS || print(programClass, "No non-copiable attributes?")) &&

            // The class to be merged into the target class must not have
            // non-copiable attributes (InnerClass, EnclosingMethod),
            // unless it is a synthetic class.
            (mergeWrapperClasses                                                 ||
             (programClass.getAccessFlags() & ClassConstants.ACC_SYNTHETIC) != 0 ||
             !hasNonCopiableAttributes(programClass)))
        {
            // We're not actually merging the classes, but only copying the
            // contents from the source class to the target class. We'll
            // then let all other classes point to it. The shrinking step
            // will finally remove the source class.
            if (DEBUG)
            {
                System.out.println("ClassMerger ["+programClass.getName()+"] -> ["+targetClass.getName()+"]");
                System.out.println("  Source interface? ["+((programClass.getAccessFlags() & ClassConstants.ACC_INTERFACE)!=0)+"]");
                System.out.println("  Target interface? ["+((targetClass.getAccessFlags() & ClassConstants.ACC_INTERFACE)!=0)+"]");
                System.out.println("  Source subclasses ["+programClass.subClasses+"]");
                System.out.println("  Target subclasses ["+targetClass.subClasses+"]");
                System.out.println("  Source superclass ["+programClass.getSuperClass().getName()+"]");
                System.out.println("  Target superclass ["+targetClass.getSuperClass().getName()+"]");

                //System.out.println("=== Before ===");
                //programClass.accept(new ClassPrinter());
                //targetClass.accept(new ClassPrinter());
            }

            // Combine the access flags.
            int targetAccessFlags = targetClass.getAccessFlags();
            int sourceAccessFlags = programClass.getAccessFlags();

            targetClass.u2accessFlags =
                ((targetAccessFlags &
                  sourceAccessFlags) &
                 (ClassConstants.ACC_INTERFACE |
                  ClassConstants.ACC_ABSTRACT)) |
                ((targetAccessFlags |
                  sourceAccessFlags) &
                 (ClassConstants.ACC_PUBLIC      |
                  ClassConstants.ACC_SUPER       |
                  ClassConstants.ACC_ANNOTATION  |
                  ClassConstants.ACC_ENUM));

            // Copy over the superclass, if it's a non-interface class being
            // merged into an interface class.
            // However, we're currently never merging in a way that changes the
            // superclass.
            //if ((programClass.getAccessFlags() & ClassConstants.ACC_INTERFACE) == 0 &&
            //    (targetClass.getAccessFlags()  & ClassConstants.ACC_INTERFACE) != 0)
            //{
            //    targetClass.u2superClass =
            //        new ConstantAdder(targetClass).addConstant(programClass, programClass.u2superClass);
            //}

            // Copy over the interfaces that aren't present yet and that
            // wouldn't cause loops in the class hierarchy.
            // Note that the code shouldn't be iterating over the original
            // list at this point. This is why we only add subclasses in
            // a separate step.
            programClass.interfaceConstantsAccept(
                new ExceptClassConstantFilter(targetClass.getName(),
                new ImplementedClassConstantFilter(targetClass,
                new ImplementingClassConstantFilter(targetClass,
                new InterfaceAdder(targetClass)))));

            // Copy over the class members.
            MemberAdder memberAdder =
                new MemberAdder(targetClass, fieldOptimizationInfoCopier);

            programClass.fieldsAccept(mergeWrapperClasses ?
                new MemberAccessFilter(ClassConstants.ACC_STATIC, 0, memberAdder) :
                memberAdder);

            programClass.methodsAccept(mergeWrapperClasses ?
                new MemberNameFilter(new NotMatcher(new FixedStringMatcher(ClassConstants.METHOD_NAME_INIT)), memberAdder) :
                memberAdder);

            // Copy over the other attributes.
            programClass.attributesAccept(
                new AttributeNameFilter(new NotMatcher(
                    new OrMatcher(new FixedStringMatcher(ClassConstants.ATTR_BootstrapMethods),
                    new OrMatcher(new FixedStringMatcher(ClassConstants.ATTR_SourceFile),
                    new OrMatcher(new FixedStringMatcher(ClassConstants.ATTR_InnerClasses),
                                  new FixedStringMatcher(ClassConstants.ATTR_EnclosingMethod))))),
                new AttributeAdder(targetClass, true)));

            // Update the optimization information of the target class.
            ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(targetClass)
                .merge(ClassOptimizationInfo.getClassOptimizationInfo(programClass));

            // Remember to replace the inlined class by the target class.
            setTargetClass(programClass, targetClass);

            //if (DEBUG)
            //{
            //    System.out.println("=== After ====");
            //    targetClass.accept(new ClassPrinter());
            //}

            // Visit the merged class, if required.
            if (extraClassVisitor != null)
            {
                extraClassVisitor.visitProgramClass(programClass);
            }
        }
    }


    public void visitLibraryClass(LibraryClass libraryClass)
    {
        // Ignore attempts to merge with a library class.
    }


    private boolean print(ProgramClass programClass, String message)
    {
        System.out.println("Merge ["+targetClass.getName()+"] <- ["+programClass.getName()+"] "+message);

        return true;
    }


    // Small utility methods.

    /**
     * Returns whether a given class is the only subclass of another given class.
     */
    private boolean isOnlySubClass(Clazz        subClass,
                                   ProgramClass clazz)
    {
        // TODO: The list of subclasses is not up to date.
        return clazz.subClasses != null     &&
               clazz.subClasses.length == 1 &&
               clazz.subClasses[0].equals(subClass);
    }


    /**
     * Returns the set of indirectly implemented interfaces.
     */
    private Set indirectlyImplementedInterfaces(Clazz clazz)
    {
        Set set = new HashSet();

        ReferencedClassVisitor referencedInterfaceCollector =
            new ReferencedClassVisitor(
            new ClassHierarchyTraveler(false, false, true, false,
            new ClassCollector(set)));

        // Visit all superclasses and  collect their interfaces.
        clazz.superClassConstantAccept(referencedInterfaceCollector);

        // Visit all interfaces and collect their interfaces.
        clazz.interfaceConstantsAccept(referencedInterfaceCollector);

        return set;
    }


    /**
     * Returns the set of interface subclasses, not including the given class.
     */
    private Set subInterfaces(Clazz clazz, Clazz exceptClass)
    {
        Set set = new HashSet();

        // Visit all subclasses, collecting the interface classes.
        clazz.hierarchyAccept(false, false, false, true,
            new ClassAccessFilter(ClassConstants.ACC_INTERFACE, 0,
            new ExceptClassesFilter(new Clazz[] { exceptClass },
            new ClassCollector(set))));

        return set;
    }


    /**
     * Returns the set of superclasses and interfaces that are initialized.
     */
    private Set sideEffectSuperClasses(Clazz clazz)
    {
        Set set = new HashSet();

        // Visit all superclasses and interfaces, collecting the ones that have
        // static initializers.
        clazz.hierarchyAccept(true, true, true, false,
                              new SideEffectClassFilter(
                              new ClassCollector(set)));

        return set;
    }


    /**
     * Returns the set of superclasses and interfaces that are used in
     * 'instanceof' tests.
     */
    private Set instanceofedSuperClasses(Clazz clazz)
    {
        Set set = new HashSet();

        // Visit all superclasses and interfaces, collecting the ones that are
        // used in an 'instanceof' test.
        clazz.hierarchyAccept(true, true, true, false,
                              new InstanceofClassFilter(
                              new ClassCollector(set)));

        return set;
    }


    /**
     * Returns the set of superclasses that are caught as exceptions.
     */
    private Set caughtSuperClasses(Clazz clazz)
    {
        // Don't bother if this isn't an exception at all.
        if (!clazz.extends_(ClassConstants.NAME_JAVA_LANG_THROWABLE))
        {
            return Collections.EMPTY_SET;
        }

        // Visit all superclasses, collecting the ones that are caught
        // (plus java.lang.Object, in the current implementation).
        Set set = new HashSet();

        clazz.hierarchyAccept(true, true, false, false,
                              new CaughtClassFilter(
                              new ClassCollector(set)));

        return set;
    }


    /**
     * Returns whether the given class has a Signature attributes containing
     * type variables or parameterized types.
     */
    private boolean hasSignatureAttribute(Clazz clazz)
    {
        AttributeCounter counter = new AttributeCounter();

        clazz.attributesAccept(
            new AttributeNameFilter(
            new FixedStringMatcher(ClassConstants.ATTR_Signature),
                counter));

        return counter.getCount() > 0;
    }

    /**
     * Returns whether the two given classes have fields with the same
     * names and descriptors.
     */
    private boolean haveAnyIdenticalFields(Clazz clazz,
                                           Clazz targetClass)
    {
        MemberCounter counter = new MemberCounter();

        // Visit all fields, counting the with the same name and descriptor in
        // the target class.
        clazz.fieldsAccept(new SimilarMemberVisitor(targetClass, true, false, false, false,
                           counter));

        return counter.getCount() > 0;
    }


    /**
     * Returns whether the given class would introduce any unwanted fields
     * in the target class.
     */
    private boolean introducesUnwantedFields(Clazz        programClass,
                                             ProgramClass targetClass)
    {
        // It's ok if the target class is never instantiated and does not
        // have any subclasses except for maybe the source class.
        if (!InstantiationClassMarker.isInstantiated(targetClass) &&
            (targetClass.subClasses == null ||
             isOnlySubClass(programClass, targetClass)))
        {
            return false;
        }

        MemberCounter counter = new MemberCounter();

        // Count all non-static fields in the the source class.
        programClass.fieldsAccept(new MemberAccessFilter(0, ClassConstants.ACC_STATIC,
                                  counter));

        return counter.getCount() > 0;
    }


    /**
     * Returns whether the given class or its subclasses shadow any fields in
     * the given target class.
     */
    private boolean shadowsAnyFields(Clazz clazz,
                                     Clazz targetClass)
    {
        MemberCounter counter = new MemberCounter();

        // Visit all fields, counting the ones that are shadowing non-private
        // fields in the class hierarchy of the target class.
        clazz.hierarchyAccept(true, false, false, true,
                              new AllFieldVisitor(
                              new SimilarMemberVisitor(targetClass, true, true, true, false,
                              new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                              counter))));

        return counter.getCount() > 0;
    }


    /**
     * Returns whether the two given classes have class members with the same
     * name and descriptor.
     */
    private boolean haveAnyIdenticalMethods(Clazz clazz,
                                            Clazz targetClass)
    {
        MemberCounter counter = new MemberCounter();

        // Visit all non-abstract methods, counting the ones that are also
        // present in the target class.
        clazz.methodsAccept(new MemberAccessFilter(0, ClassConstants.ACC_ABSTRACT,
                            new SimilarMemberVisitor(targetClass, true, false, false, false,
                            new MemberAccessFilter(0, ClassConstants.ACC_ABSTRACT,
                            counter))));

        return counter.getCount() > 0;
    }


    /**
     * Returns whether the given class would introduce any abstract methods
     * in the target class.
     */
    private boolean introducesUnwantedAbstractMethods(Clazz        clazz,
                                                      ProgramClass targetClass)
    {
        // It's ok if the target class is already abstract and does not
        // have any subclasses except for maybe the source class.
        if ((targetClass.getAccessFlags() &
             (ClassConstants.ACC_ABSTRACT |
              ClassConstants.ACC_INTERFACE)) != 0 &&
            (targetClass.subClasses == null ||
             isOnlySubClass(clazz, targetClass)))
        {
            return false;
        }

        MemberCounter counter   = new MemberCounter();
        Set           targetSet = new HashSet();

        // Collect all abstract methods, and similar abstract methods in the
        // class hierarchy of the target class.
        clazz.methodsAccept(new MemberAccessFilter(ClassConstants.ACC_ABSTRACT, 0,
                            new MultiMemberVisitor(
                                counter,

                                new SimilarMemberVisitor(targetClass, true, true, true, false,
                                new MemberAccessFilter(ClassConstants.ACC_ABSTRACT, 0,
                                new MemberCollector(false, true, true, targetSet)))
                            )));

        return targetSet.size() < counter.getCount();
    }


    /**
     * Returns whether the given class overrides any methods in the given
     * target class.
     */
    private boolean overridesAnyMethods(Clazz        clazz,
                                        ProgramClass targetClass)
    {
        // It's ok if the target class is never instantiated and does
        // not have any subclasses except for maybe the source class.
        if (!InstantiationClassMarker.isInstantiated(targetClass) &&
            (targetClass.subClasses == null ||
             isOnlySubClass(clazz, targetClass)))
        {
            return false;
        }

        MemberCounter counter = new MemberCounter();

        // Visit all non-abstract methods, counting the ones that are
        // overriding methods in the class hierarchy of the target class.
        clazz.methodsAccept(new MemberAccessFilter(0, ClassConstants.ACC_ABSTRACT,
                            new InitializerMethodFilter(null,
                            new SimilarMemberVisitor(targetClass, true, true, false, false,
                            new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE | ClassConstants.ACC_STATIC | ClassConstants.ACC_ABSTRACT,
                            counter)))));

        return counter.getCount() > 0;
    }


    /**
     * Returns whether the given class or its subclasses have private or
     * static methods that shadow any methods in the given target class.
     */
    private boolean shadowsAnyMethods(Clazz clazz,
                                      Clazz targetClass)
    {
        // It's ok if the source class already extends the target class
        // or (in practice) vice versa.
        if (clazz.extends_(targetClass) ||
            targetClass.extends_(clazz))
        {
            return false;
        }

        MemberCounter counter = new MemberCounter();

        // Visit all methods, counting the ones that are shadowing
        // final methods in the class hierarchy of the target class.
        clazz.hierarchyAccept(true, false, false, true,
                              new AllMethodVisitor(
                              new InitializerMethodFilter(null,
                              new SimilarMemberVisitor(targetClass, true, true, false, false,
                              new MemberAccessFilter(ClassConstants.ACC_FINAL, 0,
                                                     counter)))));
        if (counter.getCount() > 0)
        {
            return true;
        }

        // Visit all private methods, counting the ones that are shadowing
        // non-private methods in the class hierarchy of the target class.
        clazz.hierarchyAccept(true, false, false, true,
                              new AllMethodVisitor(
                              new MemberAccessFilter(ClassConstants.ACC_PRIVATE, 0,
                              new InitializerMethodFilter(null,
                              new SimilarMemberVisitor(targetClass, true, true, true, false,
                              new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                                                     counter))))));
        if (counter.getCount() > 0)
        {
            return true;
        }

        // Visit all static methods, counting the ones that are shadowing
        // non-private methods in the class hierarchy of the target class.
        clazz.hierarchyAccept(true, false, false, true,
                              new AllMethodVisitor(
                              new MemberAccessFilter(ClassConstants.ACC_STATIC, 0,
                              new InitializerMethodFilter(null,
                              new SimilarMemberVisitor(targetClass, true, true, true, false,
                              new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                              counter))))));

        return counter.getCount() > 0;
    }


    /**
     * Returns whether the given class has any attributes that can not be copied when
     * merging it into another class.
     */
    private boolean hasNonCopiableAttributes(Clazz clazz)
    {
        AttributeCounter counter = new AttributeCounter();

        // Copy over the other attributes.
        clazz.attributesAccept(
            new AttributeNameFilter(
                new OrMatcher(new FixedStringMatcher(ClassConstants.ATTR_InnerClasses),
                              new FixedStringMatcher(ClassConstants.ATTR_EnclosingMethod)),
            counter));

        return counter.getCount() > 0;
    }


    public static void setTargetClass(Clazz clazz, Clazz targetClass)
    {
        ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(clazz).setTargetClass(targetClass);
    }


    public static Clazz getTargetClass(Clazz clazz)
    {
        Clazz targetClass = null;

        // Return the last target class, if any.
        while (true)
        {
            clazz = ClassOptimizationInfo.getClassOptimizationInfo(clazz).getTargetClass();
            if (clazz == null)
            {
                return targetClass;
            }

            targetClass = clazz;
        }
    }


    /**
     * This MemberVisitor copies field optimization info from copied fields.
     */
    private static class FieldOptimizationInfoCopier
    extends              SimplifiedVisitor
    implements           MemberVisitor
    {
        public void visitProgramField(ProgramClass programClass, ProgramField programField)
        {
            // Copy the optimization info from the field that was just copied.
            ProgramField copiedField = (ProgramField)programField.getVisitorInfo();
            Object       info        = copiedField.getVisitorInfo();

            programField.setVisitorInfo(info instanceof ProgramFieldOptimizationInfo ?
                new ProgramFieldOptimizationInfo((ProgramFieldOptimizationInfo)info) :
                info);
        }


        public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
        {
            // Linked methods share their optimization info.
        }
    }
}
