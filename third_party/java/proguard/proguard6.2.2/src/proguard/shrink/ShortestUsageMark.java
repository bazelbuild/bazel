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
 * This class can be used as a mark when keeping classes, class members, and
 * other elements. It can be certain or preliminary. It also contains additional
 * information about the reasons why an element is being kept.
 *
 * @see ClassShrinker
 *
 * @author Eric Lafortune
 */
final class ShortestUsageMark
{
    private final boolean certain;
    private final String  reason;
    private final int     depth;
    private       Clazz   clazz;
    private       Member  member;


    /**
     * Creates a new certain ShortestUsageMark.
     * @param reason the reason for this mark.
     */
    public ShortestUsageMark(String reason)
    {
        this.certain = true;
        this.reason  = reason;
        this.depth   = 0;
    }


    /**
     * Creates a new certain ShortestUsageMark.
     * @param previousUsageMark the previous mark to which this one is linked.
     * @param reason            the reason for this mark.
     * @param clazz             the class causing this mark.
     */
    public ShortestUsageMark(ShortestUsageMark previousUsageMark,
                             String            reason,
                             int               cost,
                             Clazz             clazz)
    {
        this(previousUsageMark, reason, cost, clazz, null);
    }


    /**
     * Creates a new certain ShortestUsageMark.
     * @param previousUsageMark the previous mark to which this one is linked.
     * @param reason            the reason for this mark.
     * @param clazz             the class causing this mark.
     * @param member            the member in the above class causing this mark.
     * @param cost              the added cost of following this path.
     */
    public ShortestUsageMark(ShortestUsageMark previousUsageMark,
                             String            reason,
                             int               cost,
                             Clazz             clazz,
                             Member            member)
    {
        this.certain = true;
        this.reason  = reason;
        this.depth   = previousUsageMark.depth + cost;
        this.clazz   = clazz;
        this.member  = member;
    }


    /**
     * Creates a new ShortestUsageMark, based on another mark.
     * @param otherUsageMark the other mark, whose properties will be copied.
     * @param certain        specifies whether this is a certain mark.
     */
    public ShortestUsageMark(ShortestUsageMark otherUsageMark,
                             boolean           certain)
    {
        this.certain = certain;
        this.reason  = otherUsageMark.reason;
        this.depth   = otherUsageMark.depth;
        this.clazz   = otherUsageMark.clazz;
        this.member  = otherUsageMark.member;
    }


    /**
     * Returns whether this is a certain mark.
     */
    public boolean isCertain()
    {
        return certain;
    }


    /**
     * Returns the reason for this mark.
     */
    public String getReason()
    {
        return reason;
    }


    /**
     * Returns whether this mark has a shorter chain of reasons than the
     * given mark.
     */
    public boolean isShorter(ShortestUsageMark otherUsageMark)
    {
        return this.depth < otherUsageMark.depth;
    }


    /**
     * Returns whether this is mark is caused by the given class.
     */
    public boolean isCausedBy(Clazz clazz)
    {
        return clazz.equals(this.clazz);
    }


    /**
     * Applies the given class visitor to this mark's class, if any,
     * and if this mark doesn't have a member.
     */
    public void acceptClassVisitor(ClassVisitor classVisitor)
    {
        if (clazz  != null &&
            member == null)
        {
            clazz.accept(classVisitor);
        }
    }


    /**
     * Applies the given class visitor to this mark's member, if any.
     */
    public void acceptMemberVisitor(MemberVisitor memberVisitor)
    {
        if (clazz  != null &&
            member != null)
        {
            member.accept(clazz, memberVisitor);
        }
    }


    // Implementations for Object.

    public String toString()
    {
        return "certain=" + certain + ", depth="+depth+": " +
               reason +
               (clazz      != null ? clazz.getName() : "(none)") + ": " +
               (member     != null ? member.getName(clazz) : "(none)");
    }
}
