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
package proguard.optimize.gson;

import proguard.Configuration;
import proguard.classfile.*;
import proguard.classfile.attribute.visitor.AllAttributeVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.io.*;
import proguard.optimize.peephole.*;
import proguard.util.*;

import java.io.*;
import java.util.*;

import static proguard.classfile.ClassConstants.ACC_ENUM;
import static proguard.classfile.ClassConstants.CLASS_FILE_EXTENSION;
import static proguard.optimize.gson.GsonClassConstants.NAME_EXCLUDER;
import static proguard.optimize.gson.GsonClassConstants.NAME_GSON;
import static proguard.optimize.gson.OptimizedClassConstants.*;

/**
 * This is the entry point for the GSON optimizations.
 *
 * The optimization roughly performs the following steps:
 *
 * - Find all usages of GSON in the program code: calls to toJson() or fromJson().
 *
 * - Derive the domain classes that are involved in the GSON call, either
 *   directly (passed as argument to GSON) or indirectly (a field or element
 *   type of another domain class).
 *
 * - Inject optimized methods into the domain classes that serialize and
 *   deserialize the fields of the domain class without relying on reflection.
 *
 * - Inject and register GSON type adapters that utilize the optimized
 *   serialization and deserialization methods on the domain classes and bypass
 *   the reflective GSON implementation.
 *
 * As an additional protection measure, the JSON field names are assigned to
 * a field index. The mapping between field indices and field names is done
 * from the classes _OptimizedJsonReaderImpl and _OptimizedJsonWriterImpl, which
 * have String encryption applied to them. This allows injecting serialization
 * and deserialization code into the domain classes that have no JSON field
 * names stored in them as plain text.
 *
 * @author Lars Vandenbergh
 * @author Rob Coekaerts
 */
public class GsonOptimizer
{
    private static final boolean DEBUG = false;

    // The order of this matters to ensure that the class references are
    // initialized properly.
    private static final String[] TEMPLATE_CLASSES =
        {
            NAME_OPTIMIZED_TYPE_ADAPTER,
            NAME_GSON_UTIL,
            NAME_OPTIMIZED_JSON_READER,
            NAME_OPTIMIZED_JSON_READER_IMPL,
            NAME_OPTIMIZED_JSON_WRITER,
            NAME_OPTIMIZED_JSON_WRITER_IMPL,
            NAME_OPTIMIZED_TYPE_ADAPTER_FACTORY
        };


    /**
     * Performs the Gson optimizations.
     *
     * @param programClassPool     the program class pool on which to perform
     *                             the Gson optimizations.
     * @param libraryClassPool     the library class pool used to look up
     *                             library class references.
     * @param injectedClassNameMap the map to which injected class names are
     *                             added.
     * @param configuration        the DexGuard configuration that is applied.
     * @throws IOException         when the injected template classes can not
     *                             be read.
     */
    public void execute(ClassPool                     programClassPool,
                        ClassPool                     libraryClassPool,
                        MultiValueMap<String, String> injectedClassNameMap,
                        Configuration                 configuration) throws IOException
    {
        // Set all fields of Gson to public.
        programClassPool.classesAccept(
            new ClassNameFilter(StringUtil.join(",",
                                                NAME_GSON,
                                                NAME_EXCLUDER),
            new AllFieldVisitor(
            new MemberAccessSetter(ClassConstants.ACC_PUBLIC))));

        // To allow mocking Gson instances in unit tests, we remove the
        // final qualifier from the Gson class.
        programClassPool.classesAccept(
            new ClassNameFilter(NAME_GSON,
            new MemberAccessFlagCleaner(ClassConstants.ACC_FINAL)));

        // Setup Gson context that represents how Gson is used in program
        // class pool.
        PrintWriter out =
            new PrintWriter(System.out, true);
        WarningPrinter notePrinter =
            new WarningPrinter(out, configuration.note);

        GsonContext gsonContext = new GsonContext();
        gsonContext.setupFor(programClassPool, notePrinter);

        // Is there something to optimize at all?
        if (gsonContext.gsonDomainClassPool.size() > 0)
        {

            // Collect fields that need to be serialized and deserialized.
            OptimizedJsonInfo serializationInfo   = new OptimizedJsonInfo();
            OptimizedJsonInfo deserializationInfo = new OptimizedJsonInfo();

            OptimizedJsonFieldCollector serializedFieldCollector =
                new OptimizedJsonFieldCollector(serializationInfo,
                                                OptimizedJsonFieldCollector.Mode.serialize);
            OptimizedJsonFieldCollector deserializedFieldCollector =
                new OptimizedJsonFieldCollector(deserializationInfo,
                                                OptimizedJsonFieldCollector.Mode.deserialize);

            gsonContext.gsonDomainClassPool
                .classesAccept(
                    new MultiClassVisitor(
                        new OptimizedJsonFieldVisitor(serializedFieldCollector,
                                                      serializedFieldCollector),
                        new OptimizedJsonFieldVisitor(deserializedFieldCollector,
                                                      deserializedFieldCollector)));

            // Delete all @SerializedName and @Expose annotations
            gsonContext.gsonDomainClassPool
                .classesAccept(new GsonAnnotationCleaner(gsonContext.gsonRuntimeSettings));

            // Assign random indices to classes and fields.
            serializationInfo.assignIndices();
            deserializationInfo.assignIndices();

            // Inject all serialization and deserialization template classes.
            ClassReader helperClassReader =
                new ClassReader(false, false, false, null,
                                new MultiClassVisitor(
                                    new ClassPresenceFilter(programClassPool, null,
                                                            new ClassPoolFiller(programClassPool)),
                                    new ClassReferenceInitializer(programClassPool, libraryClassPool),
                                    new ClassSubHierarchyInitializer()));

            for (String clazz : TEMPLATE_CLASSES)
            {
                helperClassReader.read(new ClassPathDataEntry(clazz + CLASS_FILE_EXTENSION));
                injectedClassNameMap.put(GsonClassConstants.NAME_GSON,
                                         clazz);
            }

            // Inject serialization and deserialization data structures in
            // _OptimizedJsonReaderImpl and _OptimizedJsonWriterImpl.
            BranchTargetFinder branchTargetFinder = new BranchTargetFinder();
            CodeAttributeEditor codeAttributeEditor =
                new CodeAttributeEditor(true, false);

            programClassPool
                .classesAccept(NAME_OPTIMIZED_JSON_WRITER_IMPL,
                    new MultiClassVisitor(
                        new AllMemberVisitor(
                        new MemberNameFilter(OptimizedClassConstants.METHOD_NAME_INIT_NAMES,
                        new MemberDescriptorFilter(OptimizedClassConstants.METHOD_TYPE_INIT_NAMES,
                        new AllAttributeVisitor(
                        new OptimizedJsonWriterImplInitializer(programClassPool,
                                                               libraryClassPool,
                                                               codeAttributeEditor,
                                                               serializationInfo)))))));

            programClassPool
                .classesAccept(NAME_OPTIMIZED_JSON_READER_IMPL,
                    new MultiClassVisitor(
                        new AllMemberVisitor(
                        new MemberNameFilter(OptimizedClassConstants.METHOD_NAME_INIT_NAMES_MAP,
                        new MemberDescriptorFilter(OptimizedClassConstants.METHOD_TYPE_INIT_NAMES_MAP,
                        new AllAttributeVisitor(
                        new OptimizedJsonReaderImplInitializer(programClassPool,
                                                               libraryClassPool,
                                                               codeAttributeEditor,
                                                               deserializationInfo)))))));

            // Inject serialization and deserialization code in domain classes.
            gsonContext.gsonDomainClassPool
                .classesAccept(new ClassAccessFilter(0, ACC_ENUM,
                               new GsonSerializationOptimizer(programClassPool,
                                                              libraryClassPool,
                                                              gsonContext.gsonRuntimeSettings,
                                                              codeAttributeEditor,
                                                              serializationInfo,
                                                              injectedClassNameMap)));
            gsonContext.gsonDomainClassPool
                .classesAccept(new ClassAccessFilter(0, ACC_ENUM,
                               new GsonDeserializationOptimizer(programClassPool,
                                                                libraryClassPool,
                                                                gsonContext.gsonRuntimeSettings,
                                                                codeAttributeEditor,
                                                                deserializationInfo,
                                                                injectedClassNameMap)));
            gsonContext.gsonDomainClassPool
                .classesAccept(new ClassReferenceInitializer(programClassPool, libraryClassPool));

            // Inject type adapters for all serialized and deserialized classes.
            Map<String, String> typeAdapterRegistry = new HashMap<String, String>();
            OptimizedTypeAdapterAdder optimizedTypeAdapterAdder =
                new OptimizedTypeAdapterAdder(programClassPool,
                                              libraryClassPool,
                                              codeAttributeEditor,
                                              serializationInfo,
                                              deserializationInfo,
                                              injectedClassNameMap,
                                              typeAdapterRegistry,
                                              gsonContext.instanceCreatorClassPool);

            gsonContext.gsonDomainClassPool.classesAccept(optimizedTypeAdapterAdder);

            // Implement type adapter factory.
            programClassPool.classAccept(NAME_OPTIMIZED_TYPE_ADAPTER_FACTORY,
                    new MultiClassVisitor(
                        new AllMemberVisitor(
                        new AllAttributeVisitor(
                        new PeepholeOptimizer(branchTargetFinder, codeAttributeEditor,
                        new OptimizedTypeAdapterFactoryInitializer(programClassPool,
                                                                   codeAttributeEditor,
                                                                   typeAdapterRegistry,
                                                                   gsonContext.gsonRuntimeSettings)))),
                        new ClassReferenceInitializer(programClassPool, libraryClassPool)));


            // Add excluder field to Gson class if not present to support
            // @Expose in earlier Gson versions (down to 2.1).
            ProgramClass  gsonClass     = (ProgramClass) programClassPool.getClass(NAME_GSON);
            MemberCounter memberCounter = new MemberCounter();
            gsonClass.accept(new NamedFieldVisitor(FIELD_NAME_EXCLUDER,
                                                   FIELD_TYPE_EXCLUDER,
                                                   memberCounter));
            boolean addExcluder = memberCounter.getCount() == 0;
            if (addExcluder)
            {
                ConstantPoolEditor constantPoolEditor = new ConstantPoolEditor(gsonClass,
                                                                               programClassPool,
                                                                               libraryClassPool);

                int          nameIndex       = constantPoolEditor.addUtf8Constant(FIELD_NAME_EXCLUDER);
                int          descriptorIndex = constantPoolEditor.addUtf8Constant(FIELD_TYPE_EXCLUDER);
                ProgramField field           = new ProgramField(ClassConstants.ACC_PUBLIC,
                                                                nameIndex,
                                                                descriptorIndex,
                                                                null);

                ClassEditor classEditor = new ClassEditor(gsonClass);
                classEditor.addField(field);
                gsonClass.fieldsAccept(new ClassReferenceInitializer(programClassPool, libraryClassPool));
                gsonClass.constantPoolEntriesAccept(new ClassReferenceInitializer(programClassPool, libraryClassPool));
            }

            // Inject code that registers inject type adapter factory for optimized domain classes in Gson constructor.
            programClassPool.classAccept(NAME_GSON,
                    new MultiClassVisitor(
                        new AllMemberVisitor(
                            new MemberNameFilter(ClassConstants.METHOD_NAME_INIT,
                            new GsonConstructorPatcher(codeAttributeEditor, addExcluder))),
                        new ClassReferenceInitializer(programClassPool, libraryClassPool)));

            if (configuration.verbose)
            {
                System.out.println("  Number of optimized serializable classes:      " + gsonContext.gsonDomainClassPool.size() );
            }

            if (DEBUG)
            {
                // Inject instrumentation code in Gson.toJson() and Gson.fromJson().
                programClassPool.classAccept(NAME_GSON,
                    new AllMethodVisitor(
                        new MultiMemberVisitor(
                        new MemberNameFilter(GsonClassConstants.METHOD_NAME_TO_JSON,
                                             new MemberDescriptorFilter(StringUtil.join(",",
                                                                   GsonClassConstants.METHOD_TYPE_TO_JSON_OBJECT_TYPE_WRITER,
                                                                   GsonClassConstants.METHOD_TYPE_TO_JSON_JSON_ELEMENT_WRITER),
                        new AllAttributeVisitor(
                        new PeepholeOptimizer(branchTargetFinder, codeAttributeEditor,
                        new GsonInstrumentationAdder(programClassPool,
                                                     libraryClassPool,
                                                     codeAttributeEditor))))),

                        new MemberNameFilter(GsonClassConstants.METHOD_NAME_FROM_JSON,
                        new MemberDescriptorFilter(GsonClassConstants.METHOD_TYPE_FROM_JSON_JSON_READER_TYPE,
                        new AllAttributeVisitor(
                        new PeepholeOptimizer(branchTargetFinder, codeAttributeEditor,
                        new GsonInstrumentationAdder(programClassPool,
                                                     libraryClassPool,
                                                     codeAttributeEditor))))))));
            }
        }
    }
}
