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

import proguard.ClassSpecification;
import proguard.classfile.util.ClassUtil;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;


/**
 * This <code>ListPanel</code> enables the user to add, edit, move, and remove
 * ClassSpecification entries in a list.
 *
 * @author Eric Lafortune
 */
class ClassSpecificationsPanel extends ListPanel
{
    protected final ClassSpecificationDialog classSpecificationDialog;


    public ClassSpecificationsPanel(JFrame  owner,
                                    boolean includeKeepSettings,
                                    boolean includeFieldButton)
    {
        super();

        list.setCellRenderer(new MyListCellRenderer());

        classSpecificationDialog = new ClassSpecificationDialog(owner,
                                                                includeKeepSettings,
                                                                includeFieldButton);

        addAddButton();
        addEditButton();
        addRemoveButton();
        addUpButton();
        addDownButton();

        enableSelectionButtons();
    }


    protected void addAddButton()
    {
        JButton addButton = new JButton(msg("add"));
        addButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                setClassSpecification(createClassSpecification());

                int returnValue = classSpecificationDialog.showDialog();
                if (returnValue == ClassSpecificationDialog.APPROVE_OPTION)
                {
                    // Add the new element.
                    addElement(getClassSpecification());
                }
            }
        });

        addButton(tip(addButton, "addTip"));
    }


    protected void addEditButton()
    {
        JButton editButton = new JButton(msg("edit"));
        editButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                ClassSpecification selectedClassSpecification =
                    (ClassSpecification)list.getSelectedValue();

                setClassSpecification(selectedClassSpecification);
                int returnValue = classSpecificationDialog.showDialog();
                if (returnValue == ClassSpecificationDialog.APPROVE_OPTION)
                {
                    // Replace the old element.
                    setElementAt(getClassSpecification(),
                                 list.getSelectedIndex());
                }
            }
        });

        addButton(tip(editButton, "editTip"));
    }


    protected ClassSpecification createClassSpecification()
    {
        return new ClassSpecification();
    }


    protected void setClassSpecification(ClassSpecification classSpecification)
    {
        classSpecificationDialog.setClassSpecification(classSpecification);
    }


    protected ClassSpecification getClassSpecification()
    {
        return classSpecificationDialog.getClassSpecification();
    }


    /**
     * Sets the ClassSpecification objects to be represented in this panel.
     */
    public void setClassSpecifications(List classSpecifications)
    {
        listModel.clear();

        if (classSpecifications != null)
        {
            for (int index = 0; index < classSpecifications.size(); index++)
            {
                listModel.addElement(classSpecifications.get(index));
            }
        }

        // Make sure the selection buttons are properly enabled,
        // since the clear method doesn't seem to notify the listener.
        enableSelectionButtons();
    }


    /**
     * Returns the ClassSpecification objects currently represented in this panel.
     */
    public List getClassSpecifications()
    {
        int size = listModel.size();
        if (size == 0)
        {
            return null;
        }

        List classSpecifications = new ArrayList(size);
        for (int index = 0; index < size; index++)
        {
            classSpecifications.add(listModel.get(index));
        }

        return classSpecifications;
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
     * This ListCellRenderer renders ClassSpecification objects.
     */
    private class MyListCellRenderer implements ListCellRenderer
    {
        private final JLabel label = new JLabel();


        // Implementations for ListCellRenderer.

        public Component getListCellRendererComponent(JList   list,
                                                      Object  value,
                                                      int     index,
                                                      boolean isSelected,
                                                      boolean cellHasFocus)
        {
            ClassSpecification classSpecification = (ClassSpecification)value;

            label.setText(classSpecificationDialog.label(classSpecification, index));

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
}
