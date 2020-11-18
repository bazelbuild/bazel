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

/**
 * This MappingKeeper delegates all method calls to each MappingProcessor
 * in a given list.
 *
 * @author Eric Lafortune
 */
public class MultiMappingProcessor
implements   MappingProcessor
{
    private final MappingProcessor[] mappingProcessors;


    /**
     * Creates a new MultiMappingProcessor.
     * @param mappingProcessors the mapping processors to which method calls
     *                          will be delegated.
     */
    public MultiMappingProcessor(MappingProcessor[] mappingProcessors)
    {
        this.mappingProcessors = mappingProcessors;
    }


    // Implementations for MappingProcessor.

    public boolean processClassMapping(String className,
                                       String newClassName)
    {
        boolean result = false;

        for (int index = 0; index < mappingProcessors.length; index++)
        {
            result |= mappingProcessors[index].processClassMapping(className,
                                                                   newClassName);
        }

        return result;
    }


    public void processFieldMapping(String className,
                                    String fieldType,
                                    String fieldName,
                                    String newClassName,
                                    String newFieldName)
    {
        for (int index = 0; index < mappingProcessors.length; index++)
        {
            mappingProcessors[index].processFieldMapping(className,
                                                         fieldType,
                                                         fieldName,
                                                         newClassName,
                                                         newFieldName);
        }
    }


    public void processMethodMapping(String className,
                                     int    firstLineNumber,
                                     int    lastLineNumber,
                                     String methodReturnType,
                                     String methodName,
                                     String methodArguments,
                                     String newClassName,
                                     int    newFirstLineNumber,
                                     int    newLastLineNumber,
                                     String newMethodName)
    {
        for (int index = 0; index < mappingProcessors.length; index++)
        {
            mappingProcessors[index].processMethodMapping(className,
                                                          firstLineNumber,
                                                          lastLineNumber,
                                                          methodReturnType,
                                                          methodName,
                                                          methodArguments,
                                                          newClassName,
                                                          newFirstLineNumber,
                                                          newLastLineNumber,
                                                          newMethodName);
        }
    }
}
