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
package proguard.obfuscate;

import proguard.*;
import proguard.classfile.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.editor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.util.*;

import java.io.*;
import java.util.*;

/**
 * This class can perform obfuscation of class pools according to a given
 * specification.
 *
 * @author Eric Lafortune
 */
public class Obfuscator
{
    private final Configuration configuration;


    /**
     * Creates a new Obfuscator.
     */
    public Obfuscator(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Performs obfuscation of the given program class pool.
     */
    public void execute(ClassPool programClassPool,
                        ClassPool libraryClassPool) throws IOException
    {
        // Check if we have at least some keep commands.
        if (configuration.keep         == null &&
            configuration.applyMapping == null &&
            configuration.printMapping == null)
        {
            throw new IOException("You have to specify '-keep' options for the obfuscation step.");
        }

        // We're using the system's default character encoding for writing to
        // the standard output and error output.
        PrintWriter out = new PrintWriter(System.out, true);
        PrintWriter err = new PrintWriter(System.err, true);

        // Clean up any old visitor info.
        programClassPool.classesAccept(new ClassCleaner());
        libraryClassPool.classesAccept(new ClassCleaner());

        // Link all non-private, non-static methods in all class hierarchies.
        ClassVisitor memberInfoLinker =
            new BottomClassFilter(new MethodLinker());

        programClassPool.classesAccept(memberInfoLinker);
        libraryClassPool.classesAccept(memberInfoLinker);

        // If the class member names have to correspond globally,
        // additionally link all class members in all program classes.
        if (configuration.useUniqueClassMemberNames)
        {
            programClassPool.classesAccept(new AllMemberVisitor(
                                           new MethodLinker()));
        }

        // Create a visitor for marking the seeds.
        NameMarker nameMarker = new NameMarker();
        ClassPoolVisitor classPoolvisitor =
            new KeepClassSpecificationVisitorFactory(false, false, true)
                .createClassPoolVisitor(configuration.keep,
                                        nameMarker,
                                        nameMarker,
                                        nameMarker,
                                        null);
        // Mark the seeds.
        programClassPool.accept(classPoolvisitor);
        libraryClassPool.accept(classPoolvisitor);

        // All library classes and library class members keep their names.
        libraryClassPool.classesAccept(nameMarker);
        libraryClassPool.classesAccept(new AllMemberVisitor(nameMarker));

        // We also keep the names of the abstract methods of functional
        // interfaces referenced from bootstrap method arguments (additional
        // interfaces with LambdaMetafactory.altMetafactory).
        // The functional method names have to match the names in the
        // dynamic method invocations with LambdaMetafactory.
        programClassPool.classesAccept(
            new ClassVersionFilter(ClassConstants.CLASS_VERSION_1_7,
            new AllAttributeVisitor(
            new AttributeNameFilter(ClassConstants.ATTR_BootstrapMethods,
            new AllBootstrapMethodInfoVisitor(
            new AllBootstrapMethodArgumentVisitor(
            new ConstantTagFilter(ClassConstants.CONSTANT_Class,
            new ReferencedClassVisitor(
            new FunctionalInterfaceFilter(
            new ClassHierarchyTraveler(true, false, true, false,
            new AllMethodVisitor(
            new MemberAccessFilter(ClassConstants.ACC_ABSTRACT, 0,
            nameMarker))))))))))));

        // We also keep the names of the abstract methods of functional
        // interfaces that are returned by dynamic method invocations.
        // The functional method names have to match the names in the
        // dynamic method invocations with LambdaMetafactory.
        programClassPool.classesAccept(
            new ClassVersionFilter(ClassConstants.CLASS_VERSION_1_7,
            new AllConstantVisitor(
            new DynamicReturnedClassVisitor(
            new FunctionalInterfaceFilter(
            new ClassHierarchyTraveler(true, false, true, false,
            new AllMethodVisitor(
            new MemberAccessFilter(ClassConstants.ACC_ABSTRACT, 0,
            nameMarker))))))));

        // Mark attributes that have to be kept.
        AttributeVisitor attributeUsageMarker =
            new NonEmptyAttributeFilter(
            new AttributeUsageMarker());

        AttributeVisitor optionalAttributeUsageMarker =
            configuration.keepAttributes == null ? null :
                new AttributeNameFilter(configuration.keepAttributes,
                                        attributeUsageMarker);

        programClassPool.classesAccept(
            new AllAttributeVisitor(true,
            new RequiredAttributeFilter(attributeUsageMarker,
                                        optionalAttributeUsageMarker)));

        // Keep parameter names and types if specified.
        if (configuration.keepParameterNames)
        {
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new MemberNameFilter(
                new AllAttributeVisitor(true,
                new ParameterNameMarker(attributeUsageMarker)))));
        }

        // Remove the attributes that can be discarded. Note that the attributes
        // may only be discarded after the seeds have been marked, since the
        // configuration may rely on annotations.
        programClassPool.classesAccept(new AttributeShrinker());

        // Apply the mapping, if one has been specified. The mapping can
        // override the names of library classes and of library class members.
        if (configuration.applyMapping != null)
        {
            if (configuration.verbose)
            {
                out.println("Applying mapping from [" +
                            PrintWriterUtil.fileName(configuration.applyMapping) +
                            "]...");
            }

            WarningPrinter warningPrinter = new WarningPrinter(err, configuration.warn);

            MappingReader reader = new MappingReader(configuration.applyMapping);

            MappingProcessor keeper =
                new MultiMappingProcessor(new MappingProcessor[]
                {
                    new MappingKeeper(programClassPool, warningPrinter),
                    new MappingKeeper(libraryClassPool, null),
                });

            reader.pump(keeper);

            // Print out a summary of the warnings if necessary.
            int warningCount = warningPrinter.getWarningCount();
            if (warningCount > 0)
            {
                err.println("Warning: there were " + warningCount +
                            " kept classes and class members that were remapped anyway.");
                err.println("         You should adapt your configuration or edit the mapping file.");

                if (!configuration.ignoreWarnings)
                {
                    err.println("         If you are sure this remapping won't hurt,");
                    err.println("         you could try your luck using the '-ignorewarnings' option.");
                }

                err.println("         (http://proguard.sourceforge.net/manual/troubleshooting.html#mappingconflict1)");

                if (!configuration.ignoreWarnings)
                {
                    throw new IOException("Please correct the above warnings first.");
                }
            }
        }

        // Come up with new names for all classes.
        DictionaryNameFactory classNameFactory = configuration.classObfuscationDictionary != null ?
            new DictionaryNameFactory(configuration.classObfuscationDictionary, null) :
            null;

        DictionaryNameFactory packageNameFactory = configuration.packageObfuscationDictionary != null ?
            new DictionaryNameFactory(configuration.packageObfuscationDictionary, null) :
            null;

        programClassPool.classesAccept(
            new ClassObfuscator(programClassPool,
                                libraryClassPool,
                                classNameFactory,
                                packageNameFactory,
                                configuration.useMixedCaseClassNames,
                                configuration.keepPackageNames,
                                configuration.flattenPackageHierarchy,
                                configuration.repackageClasses,
                                configuration.allowAccessModification));

        // Come up with new names for all class members.
        NameFactory nameFactory = new SimpleNameFactory();
        if (configuration.obfuscationDictionary != null)
        {
            nameFactory =
                new DictionaryNameFactory(configuration.obfuscationDictionary,
                                          nameFactory);
        }

        WarningPrinter warningPrinter = new WarningPrinter(err, configuration.warn);

        // Maintain a map of names to avoid [descriptor - new name - old name].
        Map descriptorMap = new HashMap();

        // Do the class member names have to be globally unique?
        if (configuration.useUniqueClassMemberNames)
        {
            // Collect all member names in all classes.
            programClassPool.classesAccept(
                new AllMemberVisitor(
                new MemberNameCollector(configuration.overloadAggressively,
                                        descriptorMap)));

            // Assign new names to all members in all classes.
            programClassPool.classesAccept(
                new AllMemberVisitor(
                new MemberObfuscator(configuration.overloadAggressively,
                                     nameFactory,
                                     descriptorMap)));
        }
        else
        {
            // Come up with new names for all non-private class members.
            programClassPool.classesAccept(
                new MultiClassVisitor(
                    // Collect all private member names in this class and down
                    // the hierarchy.
                    new ClassHierarchyTraveler(true, false, false, true,
                    new AllMemberVisitor(
                    new MemberAccessFilter(ClassConstants.ACC_PRIVATE, 0,
                    new MemberNameCollector(configuration.overloadAggressively,
                                            descriptorMap)))),

                    // Collect all non-private member names anywhere in the
                    // hierarchy.
                    new ClassHierarchyTraveler(true, true, true, true,
                    new AllMemberVisitor(
                    new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                    new MemberNameCollector(configuration.overloadAggressively,
                                            descriptorMap)))),

                    // Assign new names to all non-private members in this class.
                    new AllMemberVisitor(
                    new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                    new MemberObfuscator(configuration.overloadAggressively,
                                         nameFactory,
                                         descriptorMap))),

                    // Clear the collected names.
                    new MapCleaner(descriptorMap)
                ));

            // Come up with new names for all private class members.
            programClassPool.classesAccept(
                new MultiClassVisitor(
                    // Collect all member names in this class.
                    new AllMemberVisitor(
                    new MemberNameCollector(configuration.overloadAggressively,
                                            descriptorMap)),

                    // Collect all non-private member names higher up the hierarchy.
                    new ClassHierarchyTraveler(false, true, true, false,
                    new AllMemberVisitor(
                    new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                    new MemberNameCollector(configuration.overloadAggressively,
                                            descriptorMap)))),

                    // Collect all member names from interfaces of abstract
                    // classes down the hierarchy.
                    // Due to an error in the JLS/JVMS, virtual invocations
                    // may end up at a private method otherwise (Sun/Oracle
                    // bugs #6691741 and #6684387, ProGuard bug #3471941,
                    // and ProGuard test #1180).
                    new ClassHierarchyTraveler(false, false, false, true,
                    new ClassAccessFilter(ClassConstants.ACC_ABSTRACT, 0,
                    new ClassHierarchyTraveler(false, false, true, false,
                    new AllMemberVisitor(
                    new MemberNameCollector(configuration.overloadAggressively,
                                            descriptorMap))))),

                    // Collect all default method names from interfaces of
                    // any classes down the hierarchy.
                    // This is an extended version of the above problem
                    // (Sun/Oracle bug #802464, ProGuard bug #662, and
                    // ProGuard test #2060).
                    new ClassHierarchyTraveler(false, false, false, true,
                    new ClassHierarchyTraveler(false, false, true, false,
                    new AllMethodVisitor(
                    new MemberAccessFilter(0, ClassConstants.ACC_ABSTRACT | ClassConstants.ACC_STATIC,
                    new MemberNameCollector(configuration.overloadAggressively,
                                            descriptorMap))))),

                    // Assign new names to all private members in this class.
                    new AllMemberVisitor(
                    new MemberAccessFilter(ClassConstants.ACC_PRIVATE, 0,
                    new MemberObfuscator(configuration.overloadAggressively,
                                         nameFactory,
                                         descriptorMap))),

                    // Clear the collected names.
                    new MapCleaner(descriptorMap)
                ));
        }

        // Some class members may have ended up with conflicting names.
        // Come up with new, globally unique names for them.
        NameFactory specialNameFactory =
            new SpecialNameFactory(new SimpleNameFactory());

        // Collect a map of special names to avoid
        // [descriptor - new name - old name].
        Map specialDescriptorMap = new HashMap();

        programClassPool.classesAccept(
            new AllMemberVisitor(
            new MemberSpecialNameFilter(
            new MemberNameCollector(configuration.overloadAggressively,
                                    specialDescriptorMap))));

        libraryClassPool.classesAccept(
            new AllMemberVisitor(
            new MemberSpecialNameFilter(
            new MemberNameCollector(configuration.overloadAggressively,
                                    specialDescriptorMap))));

        // Replace conflicting non-private member names with special names.
        programClassPool.classesAccept(
            new MultiClassVisitor(
                // Collect all private member names in this class and down
                // the hierarchy.
                new ClassHierarchyTraveler(true, false, false, true,
                new AllMemberVisitor(
                new MemberAccessFilter(ClassConstants.ACC_PRIVATE, 0,
                new MemberNameCollector(configuration.overloadAggressively,
                                        descriptorMap)))),

                // Collect all non-private member names in this class and
                // higher up the hierarchy.
                new ClassHierarchyTraveler(true, true, true, false,
                new AllMemberVisitor(
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                new MemberNameCollector(configuration.overloadAggressively,
                                        descriptorMap)))),

                // Assign new names to all conflicting non-private members
                // in this class and higher up the hierarchy.
                new ClassHierarchyTraveler(true, true, true, false,
                new AllMemberVisitor(
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                new MemberNameConflictFixer(configuration.overloadAggressively,
                                            descriptorMap,
                                            warningPrinter,
                new MemberObfuscator(configuration.overloadAggressively,
                                     specialNameFactory,
                                     specialDescriptorMap))))),

                // Clear the collected names.
                new MapCleaner(descriptorMap)
            ));

        // Replace conflicting private member names with special names.
        // This is only possible if those names were kept or mapped.
        programClassPool.classesAccept(
            new MultiClassVisitor(
                // Collect all member names in this class.
                new AllMemberVisitor(
                new MemberNameCollector(configuration.overloadAggressively,
                                        descriptorMap)),

                // Collect all non-private member names higher up the hierarchy.
                new ClassHierarchyTraveler(false, true, true, false,
                new AllMemberVisitor(
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                new MemberNameCollector(configuration.overloadAggressively,
                                        descriptorMap)))),

                // Assign new names to all conflicting private members in this
                // class.
                new AllMemberVisitor(
                new MemberAccessFilter(ClassConstants.ACC_PRIVATE, 0,
                new MemberNameConflictFixer(configuration.overloadAggressively,
                                            descriptorMap,
                                            warningPrinter,
                new MemberObfuscator(configuration.overloadAggressively,
                                     specialNameFactory,
                                     specialDescriptorMap)))),

                // Clear the collected names.
                new MapCleaner(descriptorMap)
            ));

        // Print out any warnings about member name conflicts.
        int warningCount = warningPrinter.getWarningCount();
        if (warningCount > 0)
        {
            err.println("Warning: there were " + warningCount +
                               " conflicting class member name mappings.");
            err.println("         Your configuration may be inconsistent.");

            if (!configuration.ignoreWarnings)
            {
                err.println("         If you are sure the conflicts are harmless,");
                err.println("         you could try your luck using the '-ignorewarnings' option.");
            }

            err.println("         (http://proguard.sourceforge.net/manual/troubleshooting.html#mappingconflict2)");

            if (!configuration.ignoreWarnings)
            {
                throw new IOException("Please correct the above warnings first.");
            }
        }

        // Print out the mapping, if requested.
        if (configuration.printMapping != null)
        {
            if (configuration.verbose)
            {
                out.println("Printing mapping to [" +
                            PrintWriterUtil.fileName(configuration.printMapping) +
                            "]...");
            }

            PrintWriter mappingWriter =
                PrintWriterUtil.createPrintWriter(configuration.printMapping, out);

            try
            {
                // Print out items that will be renamed.
                programClassPool.classesAcceptAlphabetically(
                    new MappingPrinter(mappingWriter));
            }
            finally
            {
                PrintWriterUtil.closePrintWriter(configuration.printMapping,
                                                 mappingWriter);
            }
        }

        if (configuration.addConfigurationDebugging)
        {
            programClassPool.classesAccept(new RenamedFlagSetter());
        }

        // Actually apply the new names.
        programClassPool.classesAccept(new ClassRenamer());
        libraryClassPool.classesAccept(new ClassRenamer());

        // Update all references to these new names.
        programClassPool.classesAccept(new ClassReferenceFixer(false));
        libraryClassPool.classesAccept(new ClassReferenceFixer(false));
        programClassPool.classesAccept(new MemberReferenceFixer(configuration.android));

        // Make package visible elements public or protected, if obfuscated
        // classes are being repackaged aggressively.
        if (configuration.repackageClasses != null &&
            configuration.allowAccessModification)
        {
            programClassPool.classesAccept(
                new AccessFixer());

            // Fix the access flags of the inner classes information.
            programClassPool.classesAccept(
                new AllAttributeVisitor(
                new AllInnerClassesInfoVisitor(
                new InnerClassesAccessFixer())));
        }

        // Fix the bridge method flags.
        programClassPool.classesAccept(
            new AllMethodVisitor(
            new BridgeMethodFixer()));

        // Rename the source file attributes, if requested.
        if (configuration.newSourceFileAttribute != null)
        {
            programClassPool.classesAccept(new SourceFileRenamer(configuration.newSourceFileAttribute));
        }

        // Remove unused constants.
        programClassPool.classesAccept(
            new ConstantPoolShrinker());
    }
}
