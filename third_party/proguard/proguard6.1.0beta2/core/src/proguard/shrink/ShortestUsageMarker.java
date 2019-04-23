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
package proguard.shrink;

import proguard.classfile.*;
import proguard.classfile.visitor.*;


/**
 * This ClassVisitor and MemberVisitor recursively marks all classes
 * and class elements that are being used. For each element, it finds the
 * shortest chain of dependencies.
 *
 * @see ClassShrinker
 *
 * @author Eric Lafortune
 */
public class ShortestUsageMarker extends UsageMarker
{
    private static final ShortestUsageMark INITIAL_MARK =
        new ShortestUsageMark("is kept by a directive in the configuration.\n\n");


    // A field acting as a parameter to the visitor methods.
    private ShortestUsageMark currentUsageMark = INITIAL_MARK;

    // A utility object to check for recursive causes.
    private final MyRecursiveCauseChecker recursiveCauseChecker = new MyRecursiveCauseChecker();


    // Overriding implementations for UsageMarker.

    protected void markProgramClassBody(ProgramClass programClass)
    {
        ShortestUsageMark previousUsageMark = currentUsageMark;

        currentUsageMark = new ShortestUsageMark(getShortestUsageMark(programClass),
                                                 "is extended by   ",
                                                 10000,
                                                 programClass);

        super.markProgramClassBody(programClass);

        currentUsageMark = previousUsageMark;
    }


    protected void markProgramFieldBody(ProgramClass programClass, ProgramField programField)
    {
        ShortestUsageMark previousUsageMark = currentUsageMark;

        currentUsageMark = new ShortestUsageMark(getShortestUsageMark(programField),
                                                 "is referenced by ",
                                                 1,
                                                 programClass,
                                                 programField);

        super.markProgramFieldBody(programClass, programField);

        currentUsageMark = previousUsageMark;
    }


    protected void markProgramMethodBody(ProgramClass programClass, ProgramMethod programMethod)
    {
        ShortestUsageMark previousUsageMark = currentUsageMark;

        currentUsageMark = new ShortestUsageMark(getShortestUsageMark(programMethod),
                                                 "is invoked by    ",
                                                 1,
                                                 programClass,
                                                 programMethod);

        super.markProgramMethodBody(programClass, programMethod);

        currentUsageMark = previousUsageMark;
    }


    protected void markMethodHierarchy(Clazz clazz, Method method)
    {
        ShortestUsageMark previousUsageMark = currentUsageMark;

        currentUsageMark = new ShortestUsageMark(getShortestUsageMark(method),
                                                 "implements       ",
                                                 100,
                                                 clazz,
                                                 method);

        super.markMethodHierarchy(clazz, method);

        currentUsageMark = previousUsageMark;
    }


    // Small utility methods.

    protected void markAsUsed(VisitorAccepter visitorAccepter)
    {
        Object visitorInfo = visitorAccepter.getVisitorInfo();

        ShortestUsageMark shortestUsageMark =
            visitorInfo != null                           &&
            visitorInfo instanceof ShortestUsageMark      &&
            !((ShortestUsageMark)visitorInfo).isCertain() &&
            !currentUsageMark.isShorter((ShortestUsageMark)visitorInfo) ?
                new ShortestUsageMark((ShortestUsageMark)visitorInfo, true):
                currentUsageMark;

        visitorAccepter.setVisitorInfo(shortestUsageMark);
    }


    protected boolean shouldBeMarkedAsUsed(VisitorAccepter visitorAccepter)
    {
        Object visitorInfo = visitorAccepter.getVisitorInfo();

        return //!(visitorAccepter instanceof Clazz &&
               //  isCausedBy(currentUsageMark, (Clazz)visitorAccepter)) &&
               (visitorInfo == null                           ||
               !(visitorInfo instanceof ShortestUsageMark)   ||
               !((ShortestUsageMark)visitorInfo).isCertain() ||
               currentUsageMark.isShorter((ShortestUsageMark)visitorInfo));
    }


    protected boolean isUsed(VisitorAccepter visitorAccepter)
    {
        Object visitorInfo = visitorAccepter.getVisitorInfo();

        return visitorInfo != null                      &&
               visitorInfo instanceof ShortestUsageMark &&
               ((ShortestUsageMark)visitorInfo).isCertain();
    }


    protected void markAsPossiblyUsed(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(new ShortestUsageMark(currentUsageMark, false));
    }


    protected boolean shouldBeMarkedAsPossiblyUsed(VisitorAccepter visitorAccepter)
    {
        Object visitorInfo = visitorAccepter.getVisitorInfo();

        return visitorInfo == null                         ||
               !(visitorInfo instanceof ShortestUsageMark) ||
               (!((ShortestUsageMark)visitorInfo).isCertain() &&
                currentUsageMark.isShorter((ShortestUsageMark)visitorInfo));
    }


    protected boolean isPossiblyUsed(VisitorAccepter visitorAccepter)
    {
        Object visitorInfo = visitorAccepter.getVisitorInfo();

        return visitorInfo != null                      &&
               visitorInfo instanceof ShortestUsageMark &&
               !((ShortestUsageMark)visitorInfo).isCertain();
    }


    protected ShortestUsageMark getShortestUsageMark(VisitorAccepter visitorAccepter)
    {
        Object visitorInfo = visitorAccepter.getVisitorInfo();

        return (ShortestUsageMark)visitorInfo;
    }


    // Small utility methods.

    private boolean isCausedBy(ShortestUsageMark shortestUsageMark,
                               Clazz             clazz)
    {
        return recursiveCauseChecker.check(shortestUsageMark, clazz);
    }


    private class MyRecursiveCauseChecker implements ClassVisitor, MemberVisitor
    {
        private Clazz   checkClass;
        private boolean isRecursing;


        public boolean check(ShortestUsageMark shortestUsageMark,
                             Clazz             clazz)
        {
            checkClass  = clazz;
            isRecursing = false;

            shortestUsageMark.acceptClassVisitor(this);
            shortestUsageMark.acceptMemberVisitor(this);

            return isRecursing;
        }

        // Implementations for ClassVisitor.

        public void visitProgramClass(ProgramClass programClass)
        {
            checkCause(programClass);
        }


        public void visitLibraryClass(LibraryClass libraryClass)
        {
            checkCause(libraryClass);
        }


        // Implementations for MemberVisitor.

        public void visitProgramField(ProgramClass programClass, ProgramField programField)
        {
            checkCause(programField);
        }


        public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
        {
            checkCause(programMethod);
        }


        public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
        {
             checkCause(libraryField);
       }


        public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
        {
            checkCause(libraryMethod);
        }


        // Small utility methods.

        private void checkCause(VisitorAccepter visitorAccepter)
        {
            if (ShortestUsageMarker.this.isUsed(visitorAccepter))
            {
                ShortestUsageMark shortestUsageMark = ShortestUsageMarker.this.getShortestUsageMark(visitorAccepter);

                // Check the class of this mark, if any
                isRecursing = shortestUsageMark.isCausedBy(checkClass);

                // Check the causing class or method, if still necessary.
                if (!isRecursing)
                {
                    shortestUsageMark.acceptClassVisitor(this);
                    shortestUsageMark.acceptMemberVisitor(this);
                }
            }
        }
    }
}
