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
package proguard.gui;

import proguard.MemberSpecification;
import proguard.classfile.ClassConstants;
import proguard.classfile.util.ClassUtil;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;


/**
 * This <code>ListPanel</code> allows the user to add, edit, move, and remove
 * MemberSpecification entries in a list.
 *
 * @author Eric Lafortune
 */
final class MemberSpecificationsPanel extends ListPanel
{
    private final MemberSpecificationDialog fieldSpecificationDialog;
    private final MemberSpecificationDialog methodSpecificationDialog;


    public MemberSpecificationsPanel(JDialog owner,
                                     boolean includeFieldButton)
    {
        super();

        super.firstSelectionButton = includeFieldButton ? 3 : 2;

        list.setCellRenderer(new MyListCellRenderer());

        fieldSpecificationDialog  = new MemberSpecificationDialog(owner, true);
        methodSpecificationDialog = new MemberSpecificationDialog(owner, false);

        if (includeFieldButton)
        {
            addAddFieldButton();
        }
        addAddMethodButton();
        addEditButton();
        addRemoveButton();
        addUpButton();
        addDownButton();

        enableSelectionButtons();
    }


    protected void addAddFieldButton()
    {
        JButton addFieldButton = new JButton(msg("addField"));
        addFieldButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                fieldSpecificationDialog.setMemberSpecification(new MemberSpecification());
                int returnValue = fieldSpecificationDialog.showDialog();
                if (returnValue == MemberSpecificationDialog.APPROVE_OPTION)
                {
                    // Add the new element.
                    addElement(new MyMemberSpecificationWrapper(fieldSpecificationDialog.getMemberSpecification(),
                                                                  true));
                }
            }
        });

        addButton(tip(addFieldButton, "addFieldTip"));
    }


    protected void addAddMethodButton()
    {
        JButton addMethodButton = new JButton(msg("addMethod"));
        addMethodButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                methodSpecificationDialog.setMemberSpecification(new MemberSpecification());
                int returnValue = methodSpecificationDialog.showDialog();
                if (returnValue == MemberSpecificationDialog.APPROVE_OPTION)
                {
                    // Add the new element.
                    addElement(new MyMemberSpecificationWrapper(methodSpecificationDialog.getMemberSpecification(),
                                                                false));
                }
            }
        });

        addButton(tip(addMethodButton, "addMethodTip"));
    }


    protected void addEditButton()
    {
        JButton editButton = new JButton(msg("edit"));
        editButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                MyMemberSpecificationWrapper wrapper =
                    (MyMemberSpecificationWrapper)list.getSelectedValue();

                MemberSpecificationDialog memberSpecificationDialog =
                    wrapper.isField ?
                        fieldSpecificationDialog :
                        methodSpecificationDialog;

                memberSpecificationDialog.setMemberSpecification(wrapper.memberSpecification);
                int returnValue = memberSpecificationDialog.showDialog();
                if (returnValue == MemberSpecificationDialog.APPROVE_OPTION)
                {
                    // Replace the old element.
                    wrapper.memberSpecification = memberSpecificationDialog.getMemberSpecification();
                    setElementAt(wrapper,
                                 list.getSelectedIndex());
                }
            }
        });

        addButton(tip(editButton, "editTip"));
    }


    /**
     * Sets the MemberSpecification instances to be represented in this panel.
     */
    public void setMemberSpecifications(List fieldSpecifications,
                                        List methodSpecifications)
    {
        listModel.clear();

        if (fieldSpecifications != null)
        {
            for (int index = 0; index < fieldSpecifications.size(); index++)
            {
                listModel.addElement(
                    new MyMemberSpecificationWrapper((MemberSpecification)fieldSpecifications.get(index),
                                                     true));
            }
        }

        if (methodSpecifications != null)
        {
            for (int index = 0; index < methodSpecifications.size(); index++)
            {
                listModel.addElement(
                    new MyMemberSpecificationWrapper((MemberSpecification)methodSpecifications.get(index),
                                                     false));
            }
        }

        // Make sure the selection buttons are properly enabled,
        // since the clear method doesn't seem to notify the listener.
        enableSelectionButtons();
    }


    /**
     * Returns the MemberSpecification instances currently represented in
     * this panel, referring to fields or to methods.
     *
     * @param isField specifies whether specifications referring to fields or
     *                specifications referring to methods should be returned.
     */
    public List getMemberSpecifications(boolean isField)
    {
        int size = listModel.size();
        if (size == 0)
        {
            return null;
        }

        List memberSpecifications = new ArrayList(size);
        for (int index = 0; index < size; index++)
        {
            MyMemberSpecificationWrapper wrapper =
                (MyMemberSpecificationWrapper)listModel.get(index);

            if (wrapper.isField == isField)
            {
                memberSpecifications.add(wrapper.memberSpecification);
            }
        }

        return memberSpecifications;
    }


    /**
     * This ListCellRenderer renders MemberSpecification objects.
     */
    private static class MyListCellRenderer implements ListCellRenderer
    {
        private final JLabel label = new JLabel();


        // Implementations for ListCellRenderer.

        public Component getListCellRendererComponent(JList   list,
                                                      Object  value,
                                                      int     index,
                                                      boolean isSelected,
                                                      boolean cellHasFocus)
        {
            MyMemberSpecificationWrapper wrapper = (MyMemberSpecificationWrapper)value;

            MemberSpecification option = wrapper.memberSpecification;
            String name       = option.name;
            String descriptor = option.descriptor;

            label.setText(wrapper.isField ?
                (descriptor == null ? name == null ?
                    "<fields>" :
                    "***" + ' ' + name :
                    ClassUtil.externalFullFieldDescription(0,
                                                           name == null ? "*" : name,
                                                           descriptor)) :
                (descriptor == null ? name == null ?
                    "<methods>" :
                    "***" + ' ' + name + "(...)" :
                    ClassUtil.externalFullMethodDescription(ClassConstants.METHOD_NAME_INIT,
                                                            0,
                                                            name == null ? "*" : name,
                                                            descriptor)));

            if (isSelected)
            {
                label.setBackground(list.getSelectionBackground());
                label.setForeground(list.getSelectionForeground());
            }
            else
            {
                label.setBackground(list.getBackground());
                label.setForeground(list.getForeground());
            }

            label.setOpaque(true);

            return label;
        }
    }


    /**
     * Attaches the tool tip from the GUI resources that corresponds to the
     * given key, to the given component.
     */
    private static JComponent tip(JComponent component, String messageKey)
    {
        component.setToolTipText(msg(messageKey));

        return component;
    }


    /**
     * Returns the message from the GUI resources that corresponds to the given
     * key.
     */
    private static String msg(String messageKey)
    {
         return GUIResources.getMessage(messageKey);
    }


    /**
     * This class wraps a MemberSpecification, additionally storing whether
     * the option refers to a field or to a method.
     */
    private static class MyMemberSpecificationWrapper
    {
        public MemberSpecification memberSpecification;
        public final boolean             isField;

        public MyMemberSpecificationWrapper(MemberSpecification memberSpecification,
                                            boolean             isField)
        {
            this.memberSpecification = memberSpecification;
            this.isField                  = isField;
        }
    }
}
